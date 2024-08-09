import re
import os
import math
import numpy as np
import pandas as pd
from scipy import constants
from scipy.stats import linregress
from scipy.signal import savgol_filter

from nemd import traj
from nemd import symbols
from nemd import molview
from nemd import logutils
from nemd import jobutils
from nemd import lammpsin
from nemd import plotutils


class Base:
    """
    The base class subclassed by analyzers.
    """

    NAME = 'base'
    DATA_EXT = '.csv'
    FIG_EXT = '.png'
    INDEX_LB = symbols.TIME_LB
    LABEL_RE = re.compile('(.*) +\((.*)\)')
    RESULTS = 'Results for '

    def __init__(self, df_reader=None, options=None, logger=None):
        """
        :param df_reader: data file reader containing structural information
        :type df_reader: 'nemd.oplsua.DataFileReader'
        :param options: the options from command line
        :type options: 'argparse.Namespace'
        :param logger: the logger to log messages
        :type logger: 'logging.Logger'
        """
        self.df_reader = df_reader
        self.options = options
        self.logger = logger
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

    def saveData(self, float_format='%.4g'):
        """
        Save the data.

        :param float_format str: the format to save float
        """
        if self.data.empty:
            return
        self.data.to_csv(self.outfile, float_format=float_format)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')

    def fit(self):
        """
        Select the data and report average with std.

        :return int, int: the start and end index for the selected data
        """
        if self.data.empty:
            return
        if self.sidx is None:
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

    def plot(self, marker_num=10, use_column=False):
        """
        Plot and save the data (interactively).

        :param marker_num: add markers when the number of points equals or is
            less than this value
        :type marker_num: int
        :param use_column: use column label in output filename
        :type use_column: bool
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
                gdata = self.data.iloc[[self.sidx, self.eidx]]
                ax.plot(gdata.index, gdata.iloc[:, 0], '.-g')
            label, unit, _ = self.parseLabel(self.data.index.name)
            ax.set_xlabel(f"{label} ({unit})")
            ax.set_ylabel(self.data.columns.values.tolist()[0])
            fname = f"{self.options.jobname}_{self.NAME}{self.FIG_EXT}"
            if use_column:
                name = self.data.columns[0].split('(')[0].strip().lower()
                fname = f"{self.options.jobname}_{name}_{self.NAME}{self.FIG_EXT}"
            file_path = os.path.join(os.path.dirname(self.outfile), fname)
            fig.savefig(file_path)
            jobutils.add_outfile(file_path, jobname=self.options.jobname)
        self.log(f'{self.DESCR.capitalize()} figure saved as {file_path}')

    def log(self, msg):
        """
        Print this message into the log file as information.

        :param msg str: the msg to be printed
        """
        if self.logger:
            logutils.log(self.logger, msg)
        else:
            print(msg)

    def log_debug(self, msg):
        """
        Print this message into the log file in debug mode.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def log_warning(self, msg):
        """
        Print this warning message into log file.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.warning(msg)
        else:
            print(msg)


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

    @classmethod
    def fit(cls, data, log=None):
        """
        Smooth the rdf data and report peaks.

        :param data: distance vs count
        :type data: 'pandas.core.frame.DataFrame'
        :param log: the function to print user-facing information
        :type log: 'function'
        """
        if data.empty:
            return
        raveled = np.ravel(data[data.columns[0]])
        smoothed = savgol_filter(raveled, window_length=31, polyorder=2)
        row = data.iloc[smoothed.argmax()]
        log(f'Peak position: {row.name}; peak value: {row.values[0]: .2f}')


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

    def fit(self, spct=0.1, epct=0.2, log=None):
        """
        Select and fit the mean squared displacement to calculate the diffusion
        coefficient.

        :param spct float: exclude the frames of this percentage at head
        :param epct float: exclude the frames of this percentage at tail
        :param log 'function': the function to print user-facing information
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
        log(f'{slope/6:.4g} {symbols.PLUS_MIN} {std_err/6:.4g} cm^2/s'
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
    THERMO = 'thermo'
    TEMP = 'Temp'
    EPAIR = 'E_pair'
    E_MOL = 'E_mol'
    TOTENG = 'TotEng'
    PRESS = 'Press'
    VOLUME = 'Volume'
    TASKS = [TEMP, EPAIR, E_MOL, TOTENG, PRESS, VOLUME]

    def __init__(self, thermo=None, task=None, **kwargs):
        """
        :param thermo: the thermodynamic data
        :type thermo: 'pandas.core.frame.DataFrame'
        :param task: the thermodynamic task name
        :type task: str
        """
        super().__init__(**kwargs)
        self.thermo = thermo
        self.task = task
        self.NAME = self.task.lower()
        self.DESCR = self.task.capitalize()

    def setData(self):
        """
        Select data by the thermo task name.
        """
        if self.data is not None:
            return
        column_re = re.compile(f"{self.task} +\((.*)\)")
        column = [x for x in self.thermo.columns if column_re.match(x)][0]
        self.data = self.thermo[column].to_frame()


ANALYZER = [Density, RDF, MSD, Clash, View, XYZ]
ANALYZER = {getattr(x, 'NAME'): x for x in ANALYZER}
NO_COMBINE = [View, XYZ]
