import re
import os
import math
import numpy as np
import pandas as pd
from scipy import constants
from types import SimpleNamespace
from scipy.stats import linregress
from scipy.signal import savgol_filter

from nemd import traj
from nemd import symbols
from nemd import molview
from nemd import logutils
from nemd import jobutils
from nemd import lammpsin
from nemd import plotutils


class Base(logutils.Base):
    """
    The base class subclassed by analyzers.
    """

    NAME = 'base'
    DATA_EXT = '.csv'
    FIG_EXT = '.png'
    FLOAT_FMT = '%.4g'
    LABEL_RE = re.compile('(.*) +\((.*)\)')

    def __init__(self, df_reader=None, options=None, logger=None):
        """
        :param df_reader: data file reader containing structural information
        :type df_reader: 'nemd.oplsua.DataFileReader'
        :param options: the options from command line
        :type options: 'argparse.Namespace'
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.df_reader = df_reader
        self.options = options
        self.data = None
        self.idx = 0
        self.sidx = None
        self.eidx = None
        self.result = None
        self.outfile = self.getFilename(self.options)
        jobutils.add_outfile(self.outfile, jobname=self.options.jobname)

    @classmethod
    def getFilename(cls, options):
        """
        :param options: the options from command line
        :type options: 'argparse.Namespace' or str
        :return str: the filename of the data file.
        """
        if isinstance(options, str):
            return f"{options}_{cls.NAME}{cls.DATA_EXT}"
        if not hasattr(options, 'jobs'):
            return f"{options.jobname}_{cls.NAME}{cls.DATA_EXT}"
        filename = f"{options.jobname}_{cls.NAME}_{options.id}{cls.DATA_EXT}"
        return os.path.join(options.dir, filename)

    def run(self):
        """
        Main method to run the analyzer.
        """
        self.readData()
        self.setData()
        self.saveData()
        self.fit()
        self.plot()

    def readData(self):
        """
        Read the output files from independent runs to set the data.
        """
        if not hasattr(self.options, 'jobs'):
            return
        filename = self.getFilename(self.options.name)
        files = [x.fn(filename) for x in self.options.jobs]
        datas = [pd.read_csv(x, index_col=0) for x in files]
        # 'Time (ps)': None; 'Time (ps) (0)': '0'; 'Time (ps) (0, 1)': '0, 1'
        names = [self.parseIndexName(x.index.name) for x in datas]
        if len(datas) == 1:
            # One single run
            self.data = datas[0]
            self.sidx, self.eidx = names[0][-2:]
            return
        # Runs with different randomized seeds are truncated backwards
        num = min([x.shape[0] for x in datas])
        datas = [x.iloc[-num:] for x in datas]
        # Averaged index
        indexes = [x.index.to_numpy().reshape(-1, 1) for x in datas]
        index_ave = np.concatenate(indexes, axis=1).mean(axis=1)
        label, unit, _, _ = names[0]
        _, _, self.sidx, self.eidx = pd.DataFrame(names).min()
        name = f"{label} ({unit}) ({self.sidx} {self.eidx})"
        index = pd.Index(index_ave, name=name)
        # Averaged value and standard deviation
        vals = [x.iloc[:, 0].to_numpy().reshape(-1, 1) for x in datas]
        vals = np.concatenate(vals, axis=1)
        mean_lb = f'{datas[0].columns[0]} (num={vals.shape[-1]})'
        sd_lb = symbols.SD_PREFIX + mean_lb
        data = {mean_lb: vals.mean(axis=1), sd_lb: vals.std(axis=1)}
        self.data = pd.DataFrame(data, index=index)

    def setData(self):
        """
        Set the data.
        """
        pass

    def saveData(self):
        """
        Save the data.
        """
        if self.data.empty:
            return
        self.data.to_csv(self.outfile, float_format=self.FLOAT_FMT)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')

    def fit(self):
        """
        Select the data and report average with std.

        :return int, int: the start and end index for the selected data
        """
        if self.data.empty:
            return
        if self.sidx is None:
            # idx is the index set based on the last ptc and input data shape
            self.sidx = self.idx
        sel = self.data.iloc[self.sidx:self.eidx]
        data_lb = sel.columns[0]
        ave = sel[data_lb].mean()
        sd_lb = None if len(sel.columns) == 1 else sel.columns[1]
        sd = sel[data_lb].std() if sd_lb is None else sel[sd_lb].mean()
        if sd_lb is None:
            sd_lb = symbols.SD_PREFIX + data_lb
        self.result = pd.Series({data_lb: ave, sd_lb: sd})
        self.result.index.name = sel.index.name
        label, unit, _ = self.parseLabel(self.data.columns[0])
        stime, etime = sel.index[0], sel.index[-1]
        self.log(f'{label}: {ave:.4g} {symbols.PLUS_MIN} {sd:.4g} {unit} '
                 f'{symbols.ELEMENT_OF} [{stime:.4f}, {etime:.4f}] ps')

    def parseIndexName(self, name, sidx=0, eidx=None):
        """
        Parse the index name to get the label, unit, start index and end index.

        :param name: the column name
        return str, str, int, int: label, unit, start index, and end index.
        """
        label, unit, other = self.parseLabel(name)
        if other is None:
            return label, unit, sidx, eidx
        splitted = list(map(int, other.split()))
        sidx = splitted[0]
        if len(splitted) > 2:
            eidx = splitted[1]
        return label, unit, sidx, eidx

    @classmethod
    def parseLabel(cls, name):
        """
        Get the label and unit from the column name.

        :param name: the column name
        return str, str: the label and unit
        """
        # 'Density (g/cm^3)
        (label, unit), other = cls.LABEL_RE.match(name).groups(), None
        match = cls.LABEL_RE.match(label)
        if match:
            # 'Density (g/cm^3) (num=4)' as data.columns[0]
            (label, unit), other = match.groups(), unit
        return label, unit, other

    def plot(self, marker_num=10):
        """
        Plot and save the data (interactively).

        :param marker_num: add markers when the number of points equals or is
            less than this value
        :type marker_num: int
        """
        if self.data.empty:
            return
        with plotutils.get_pyplot(inav=self.options.interactive,
                                  name=self.DESCR.upper()) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.13, 0.1, 0.8, 0.8])
            line_style = '--' if any([self.sidx, self.eidx]) else '-'
            if len(self.data) < marker_num:
                line_style += '*'
            ax.plot(self.data.index,
                    self.data.iloc[:, 0],
                    line_style,
                    label='average')
            if self.data.shape[-1] == 2 and self.data.iloc[:, 1].any():
                # Data has non-zero standard deviation column
                vals, errors = self.data.iloc[:, 0], self.data.iloc[:, 1]
                ax.fill_between(self.data.index,
                                vals - errors,
                                vals + errors,
                                color='y',
                                label='stdev',
                                alpha=0.3)
                ax.legend()
            if any([self.sidx, self.eidx]):
                gdata = self.data.iloc[self.sidx:self.eidx]
                ax.plot(gdata.index, gdata.iloc[:, 0], '.-g')
            label, unit, _ = self.parseLabel(self.data.index.name)
            ax.set_xlabel(f"{label} ({unit})")
            ax.set_ylabel(self.data.columns.values.tolist()[0])
            pathname = self.outfile[:-len(self.DATA_EXT)] + self.FIG_EXT
            fig.savefig(pathname)
            jobutils.add_outfile(pathname, jobname=self.options.jobname)
        self.log(f'{self.DESCR.capitalize()} figure saved as {pathname}')


class TrajBase(Base):
    """
    The base class for trajectory analyzers.
    """

    def __init__(self, time=None, frms=None, gids=None, **kwargs):
        """
        :param time: time array
        :type time: 'numpy.ndarray'
        :param frms: traj frames
        :type frms: list of 'nemd.traj.Frame'
        :param gids: global ids for the selected atom
        :type gids: list of int
        """
        super().__init__(**kwargs)
        self.time = time
        self.frms = frms
        self.gids = gids
        if self.time is not None:
            self.idx = int(self.LABEL_RE.match(self.time.name).groups()[1])


class Density(TrajBase):
    """
    The density analyzer.
    """

    NAME = 'density'
    DESCR = NAME
    PNAME = NAME.capitalize()
    UNIT = 'g/cm^3'
    LABEL = f'{PNAME} ({UNIT})'

    def setData(self):
        """
        Set the time vs density data.
        """
        if self.data is not None:
            return
        mass = self.df_reader.molecular_weight / constants.Avogadro
        mass_scaled = mass / (constants.angstrom / constants.centi)**3
        data = [mass_scaled / x.volume for x in self.frms]
        self.data = pd.DataFrame({self.LABEL: data}, index=self.time)


class RDF(TrajBase):
    """
    The radial distribution function analyzer.
    """

    NAME = 'rdf'
    DESCR = 'radial distribution function'
    PNAME = 'g'
    UNIT = 'r'
    LABEL = f'{PNAME} ({UNIT})'
    INDEX_LB = f'r ({symbols.ANGSTROM})'
    DEFAULT_CUT = lammpsin.In.DEFAULT_CUT

    def setData(self, res=0.02, dcut=None):
        """
        Set the radial distribution function.

        :param res float: the rdf minimum step
        :param dcut float: the cutoff distance to look for neighbors. If None,
            all the neighbors are counted when the cell is not significantly
             larger than the LJ cutoff.
        """
        if self.data is not None:
            return
        if len(self.gids) < 2:
            self.log_warning("RDF requires least two atoms selected.")
            self.data = pd.DataFrame(data={self.LABEL: []})
            return

        frms = self.frms[self.idx:]
        span = np.array([x.box.span for x in frms])
        vol = span.prod(axis=1)
        self.log(f'The volume fluctuates: [{vol.min():.2f} {vol.max():.2f}] '
                 f'{symbols.ANGSTROM}^3')

        mdist, dcell = span.min() * 0.5, None
        # The auto resolution based on cut grabs left, middle, and right boxes
        if dcut is None and mdist > self.DEFAULT_CUT * 2.5:
            # Cell is significant larger than LJ cut off, and thus use LJ cut
            dcell = traj.DistanceCell(gids=self.gids, cut=self.DEFAULT_CUT)
            self.log(f"Only neighbors within {dcut} are accurate.")
            mini_res = span.min() / traj.DistanceCell.GRID_MAX
            mdist = max(self.DEFAULT_CUT, mini_res)

        res = min(res, mdist / 100)
        bins = round(mdist / res)
        hist_range = [res / 2, res * bins + res / 2]
        rdf, num = np.zeros((bins)), len(self.gids)
        tenth, threshold, = len(frms) / 10., 0
        for idx, frm in enumerate(frms, start=1):
            self.log_debug(f"Analyzing frame {idx} for RDF..")
            if dcell is None:
                dists = frm.pairDists(grp1=self.gids)
            else:
                dcell.setup(frm)
                dists = dcell.pairDists()
            hist, edge = np.histogram(dists, range=hist_range, bins=bins)
            mid = np.array([x for x in zip(edge[:-1], edge[1:])]).mean(axis=1)
            # 4pi*r^2*dr*rho from Radial distribution function - Wikipedia
            norm_factor = 4 * np.pi * mid**2 * res * num / frm.volume
            # Stands at every id but either (1->2) or (2->1) is computed
            rdf += (hist * 2 / num / norm_factor)
            if idx >= threshold:
                new_line = "" if idx == len(frms) else ", [!n]"
                self.log(f"{int(idx / len(frms) * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)
        rdf /= len(frms)
        mid, rdf = np.concatenate(([0], mid)), np.concatenate(([0], rdf))
        index = pd.Index(data=mid, name=self.INDEX_LB)
        self.data = pd.DataFrame(data={self.LABEL: rdf}, index=index)

    def fit(self):
        """
        Smooth the rdf data and report peaks.
        """
        if self.data.empty:
            return
        raveled = np.ravel(self.data.iloc[:, 0])
        smoothed = savgol_filter(raveled, window_length=31, polyorder=2)
        row = self.data.iloc[smoothed.argmax()]
        self.log(
            f'Peak position: {row.name}; peak value: {row.values[0]: .2f}')


class MSD(TrajBase):
    """
    The mean squared displacement analyzer.
    """

    NAME = 'msd'
    DESCR = 'mean squared displacement'
    PNAME = NAME.upper()
    UNIT = f'{symbols.ANGSTROM}^2'
    LABEL = f'{PNAME} ({UNIT})'
    INDEX_LB = 'Tau (ps)'

    def setData(self):
        """
        Set the mean squared displacement and diffusion coefficient.
        """
        if self.data is not None:
            return
        if not self.gids:
            self.log_warning("No atoms selected for MSD.")
            self.data = pd.DataFrame({self.LABEL: []})
            return
        gids = list(self.gids)
        masses = self.df_reader.masses.mass[self.df_reader.atoms.type_id[gids]]
        frms = self.frms[self.idx:]
        msd, num = [0], len(frms)
        for idx in range(1, num):
            disp = [
                x.xyz[gids, :] - y.xyz[gids, :]
                for x, y in zip(frms[idx:], frms[:-idx])
            ]
            data = np.array([np.linalg.norm(x, axis=1) for x in disp])
            sdata = np.square(data)
            msd.append(np.average(sdata.mean(axis=0), weights=masses))
        ps_time = self.time[self.idx:][:num]
        tau_idx = pd.Index(data=ps_time - ps_time[0], name=self.INDEX_LB)
        self.data = pd.DataFrame({self.LABEL: msd}, index=tau_idx)

    def fit(self, spct=0.1, epct=0.2):
        """
        Select and fit the mean squared displacement to calculate the diffusion
        coefficient.

        :param spct float: exclude the frames of this percentage at head
        :param epct float: exclude the frames of this percentage at tail
        :return int, int: the start and end index for the selected data
        """
        if self.data.empty:
            return
        num = self.data.shape[0]
        if self.sidx is None:
            self.sidx = math.floor(num * spct)
            self.eidx = math.ceil(num * (1 - epct))
        sel = self.data.iloc[self.sidx:self.eidx]
        # Standard error of the slope, under the assumption of residual normality
        xvals = sel.index * constants.pico
        yvals = sel.iloc[:, 0] * (constants.angstrom / constants.centi)**2
        slope, intercept, rvalue, p_value, std_err = linregress(xvals, yvals)
        # MSD=2nDt https://en.wikipedia.org/wiki/Mean_squared_displacement
        self.log(f'{slope/6:.4g} {symbols.PLUS_MIN} {std_err/6:.4g} cm^2/s'
                 f' (R-squared: {rvalue**2:.4f}) linear fit of'
                 f' [{sel.index[0]:.4f} {sel.index[-1]:.4f}] ps')


class Clash(TrajBase):
    """
    The clash analyzer.
    """

    NAME = 'clash'
    DESCR = 'clash count'
    PNAME = NAME.capitalize()
    UNIT = 'count'
    LABEL = f'{PNAME} ({UNIT})'

    def setData(self):
        """
        Set the time vs clash number.
        """
        if self.data is not None:
            return
        if not self.gids:
            self.log_warning("No atoms selected for clash counting.")
            self.data = pd.DataFrame({self.LABEL: []})
            return
        dcell = traj.DistanceCell(gids=set(self.gids), struct=self.df_reader)
        data = []
        for frm in self.frms:
            dcell.setup(frm)
            data.append(len(dcell.getClashes()))
        self.data = pd.DataFrame(data={self.LABEL: data}, index=self.time)


class XYZ(TrajBase):
    """
    The XYZ coordinate converter.
    """

    NAME = 'xyz'
    DESCR = NAME.upper()
    DATA_EXT = '.xyz'

    def run(self, wrapped=True, broken_bonds=False, glue=False):
        """
        Write the coordinates of the trajectory into XYZ format.

        :param wrapped bool: coordinates are wrapped into the PBC box.
        :param bond_across_pbc bool: allow bonds passing PBC boundaries.
        :param glue bool: circular mean to compact the molecules.

        NOTE: wrapped=False & glue=False is good for diffusion virtualization
        wrapped True & broken_bonds=False is good for box fully filled with molecules
        broken_bonds=False & glue=True is good for molecules droplets in vacuum
        Not all combination make physical senses.
        """
        with open(self.outfile, 'w') as self.out_fh:
            # XYZ analyzer may change the coordinates
            for frm in self.frms:
                if wrapped:
                    frm.wrapCoords(broken_bonds, dreader=self.df_reader)
                if glue:
                    frm.glue(dreader=self.df_reader)
                frm.write(self.out_fh, dreader=self.df_reader)
        self.log(f"{self.DESCR} coordinates are written into {self.outfile}")


class View(TrajBase):
    """
    The coordinate visualizer.
    """

    NAME = 'view'
    DESCR = 'trajectory visualization'
    DATA_EXT = '.html'

    def run(self):
        """
        Main method to run the visualization.
        """
        frm_vw = molview.FrameView(df_reader=self.df_reader)
        frm_vw.setData(self.frms[0])
        frm_vw.setElements()
        frm_vw.addTraces()
        frm_vw.setFrames(self.frms)
        frm_vw.updateLayout()
        frm_vw.show(outfile=self.outfile, inav=self.options.interactive)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')


class Thermo(Base):

    NAME = 'thermo'
    DESCR = 'Thermodynamic information'
    FLOAT_FMT = '%.8g'
    THERMO = 'thermo'
    TEMP = 'temp'
    EPAIR = 'e_pair'
    E_MOL = 'e_mol'
    TOTENG = 'toteng'
    PRESS = 'press'
    VOLUME = 'volume'
    TASKS = [TEMP, EPAIR, E_MOL, TOTENG, PRESS, VOLUME]

    def __init__(self, thermo=None, **kwargs):
        """
        :param thermo: the thermodynamic data
        :type thermo: 'pandas.core.frame.DataFrame'
        """
        super().__init__(**kwargs)
        self.thermo = thermo
        if self.thermo is not None:
            self.idx = int(
                self.LABEL_RE.match(self.thermo.index.name).groups()[1])

    def setData(self):
        """
        Select data by the thermo task name.
        """
        if self.data is not None:
            return
        column_re = re.compile(f"{self.NAME} +\((.*)\)", re.IGNORECASE)
        column = [x for x in self.thermo.columns if column_re.match(x)][0]
        self.data = self.thermo[column].to_frame()


ANALYZER = [Density, RDF, MSD, Clash, View, XYZ]
for name in Thermo.TASKS:
    ANALYZER.append(
        type(name, (Thermo, ), {
            'NAME': name,
            'DESCR': name.capitalize()
        }))
ANALYZER = {getattr(x, 'NAME'): x for x in ANALYZER}


class Agg(logutils.Base):

    DATA_EXT = '.csv'
    FIG_EXT = '.png'

    def __init__(self, task=None, jobs=None, options=None, logger=None):
        """
        :param task str: the task name to analyze
        :param jobs: state point parameters, grouped jobs
        :type jobs: list of (pandas.Series, 'signac.job.Job') tuples
        :param options 'argparse.Namespace': the options from command line
        :param logger 'logging.Logger': the logger to log messages
        """
        super().__init__(logger=logger)
        self.task = task
        self.jobs = jobs
        self.options = options
        self.Anlz = None
        self.result = pd.DataFrame()
        self.yvals = None
        self.ydevs = None
        self.xvals = None
        self.outfile = f"{self.options.jobname}_{self.task}{self.DATA_EXT}"

    def run(self):
        """
        Main method to aggregate the analyzer output files over all parameters.
        """
        self.setAnalyzer()
        self.setResults()
        self.save()
        self.setVals()
        self.fit()
        self.plot()

    def setAnalyzer(self):
        """
        Set the analyzer class for the given task.
        """
        self.Anlz = ANALYZER.get(self.task)
        if self.Anlz is None:
            self.log(f"Aggregator Analyzer not found for task {self.task}")

    def setResults(self):
        """
        Set results for the given task over grouped jobs.
        """
        if self.Anlz is None:
            return
        self.log(f"Aggregation Task: {self.task}")
        shared = vars(self.options)
        if self.options.interactive:
            shared = shared.copy()
            shared['interactive'] = len(self.jobs) > 1
        for parm, jobs in self.jobs:
            if not parm.empty:
                pstr = parm.to_csv(lineterminator=' ', sep='=', header=False)
                self.log(f"Aggregation Parameters (num={len(jobs)}): {pstr}")
            options = SimpleNamespace(**shared, id=parm.index.name, jobs=jobs)
            anlz = self.Anlz(options=options, logger=self.logger)
            anlz.run()
            if anlz.result is None:
                continue
            result = [anlz.result] if parm.empty else [parm, anlz.result]
            result = pd.concat(result).to_frame().transpose()
            self.result = pd.concat([self.result, result])

    def save(self):
        """
        Save the results to a file.
        """
        if self.result.empty:
            return

        self.result.to_csv(self.outfile, index=False)
        task = self.task.capitalize()
        self.log(f"{task} of all parameters saved to {self.outfile}")
        jobutils.add_outfile(self.outfile, jobname=self.options.jobname)

    def setVals(self):
        """
        Set the x, y, and y standard deviation from the results.
        """
        y_lb_re = re.compile(f"{self.task} +\((.*)\)", re.IGNORECASE)
        y_sd_lb_re = f"{symbols.SD_PREFIX}{self.task} +\((.*)\)"
        y_sd_lb_re = re.compile(y_sd_lb_re, re.IGNORECASE)
        y_lb = [x for x in self.result.columns if y_lb_re.match(x)]
        self.yvals = self.result[y_lb].iloc[:, 0]
        ysd_lb = [x for x in self.result.columns if y_sd_lb_re.match(x)]
        self.ydevs = self.result[ysd_lb].iloc[:, 0]
        x_lbs = list(set(self.result.columns).difference(y_lb + ysd_lb))
        self.xvals = self.result[x_lbs]
        rename = {
            x: ' '.join([y.capitalize() for y in x.split('_')])
            for x in self.xvals.columns
        }
        self.xvals = self.xvals.rename(columns=rename)

    def fit(self):
        """
        Fit the data and report.
        """
        if self.xvals.empty or self.xvals.size == 1:
            return
        index = self.yvals.argmin()
        self.log(f"The minimum {self.yvals.name} of {self.yvals.iloc[index]} "
                 f"is found with the {self.xvals.columns[0].replace('_',' ')} "
                 f"being {self.xvals.iloc[index, 0]}")

    def plot(self, xtick_num=12):
        """
        Plot the results.

        :param xtick_num int: the maximum number of xticks to show
        """
        if self.xvals.empty:
            return
        with plotutils.get_pyplot(inav=self.options.interactive,
                                  name=self.task.upper()) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.13, 0.1, 0.8, 0.8])
            ax.plot(self.xvals.iloc[:, 0], self.yvals, '--*', label='average')
            if not self.ydevs.isnull().any():
                # Data has non-zero standard deviation column
                ax.fill_between(self.xvals,
                                self.yvals - self.ydevs,
                                self.yvals + self.ydevs,
                                color='y',
                                label='stdev',
                                alpha=0.3)
                ax.legend()
            ax.set_xlabel(self.xvals.columns[0])
            if self.xvals.iloc[:, 0].size > xtick_num:
                intvl = round(self.xvals.iloc[:, 0].size / xtick_num)
                ax.set_xticks(self.xvals.iloc[:, 0].values[::intvl])
            ax.set_ylabel(self.yvals.name)
            pathname = self.outfile[:-len(self.DATA_EXT)] + self.FIG_EXT
            fig.savefig(pathname)
            jobutils.add_outfile(pathname, jobname=self.options.jobname)
        self.log(f'{self.task.upper()} figure saved as {pathname}')
