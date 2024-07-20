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
    TIME_LB = 'Time ps'
    TIME_RE = f"(?<={TIME_LB} \().+(?=\))"
    ILABEL = TIME_LB
    RESULTS = 'Results for '

    def __init__(self,
                 time,
                 frms,
                 sidx=None,
                 df_reader=None,
                 gids=None,
                 options=None,
                 logger=None):
        """
        :param time: time array
        :type time: 'numpy.ndarray'
        :param frms: traj frames
        :type frms: list of 'nemd.traj.Frame'
        :param sidx: the starting frame index after excluding the first xxx pct
        :type sidx: int
        :param df_reader: data file reader containing structural information
        :type df_reader: 'nemd.oplsua.DataFileReader'
        :param gids: global ids for the selected atom
        :type gids: list of int
        :param options: parsed commandline options
        :type options: 'argparse.Namespace'
        :param logger: the handle to print info
        :type logger: 'logging.Logger'
        """
        self.time = time
        self.frms = frms
        self.sidx = sidx
        self.df_reader = df_reader
        self.gids = gids
        self.options = options
        self.logger = logger
        self.data = None
        self.outfile = f"{self.options.jobname}_{self.NAME}{self.DATA_EXT}"

    def run(self):
        """
        Main method to run the analyzer.
        """
        self.setData()
        self.saveData()
        sidx, eidx = self.fit(self.data, log=self.log)
        self.plot(self.data,
                  inav=self.options.interactive,
                  sidx=sidx,
                  eidx=eidx,
                  name=self.options.jobname,
                  log=self.log)

    def setData(self):
        """
        Set the data. Must be over-written by the subclass.
        """
        raise NotImplemented

    def saveData(self, float_format='%.4g'):
        """
        Save the data.

        :param float_format str: the format to save float
        """
        if self.data.empty:
            return
        self.data.to_csv(self.outfile, float_format=float_format)
        jobutils.add_outfile(self.outfile, jobname=self.options.jobname)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')

    @classmethod
    def fit(cls, data, log=None):
        """
        Select the data and report average with std.

        :param data 'pandas.core.frame.DataFrame': time vs data
        :param log 'function': the function to print user-facing information
        :return int, int: the start and end index for the selected data
        """
        if data.empty:
            return 0, None
        sidx = int(re.findall(cls.TIME_RE, data.index.name)[0])
        sel = data.iloc[sidx:]
        ave = sel.mean().iloc[0]
        std = sel.std().iloc[0] if sel.shape[1] == 1 else sel.mean().iloc[1]
        log(f'{ave:.4g} {symbols.PLUS_MIN} {std:.4g} {cls.UNIT} '
            f'{symbols.ELEMENT_OF} [{data.index[sidx]:.4f}, '
            f'{data.index[-1]:.4f}] ps')
        return sidx, None

    @classmethod
    def plot(cls, data, name, sidx=None, eidx=None, log=None, inav=False):
        """
        Plot and save the data (interactively).

        :param data: data to plot
        :type data: 'pandas.core.frame.DataFrame'
        :param name: the taskname based on which output file is set
        :type name: str
        :param sidx: the starting index when selecting data
        :type sidx: int
        :param eidx: the ending index when selecting data
        :type eidx: int
        :param log: the function to print user-facing information
        :type log: 'function'
        :param inav: pop up window and show plot during code execution if
            interactive mode is on
        :type inav: bool
        """
        if data.empty:
            return
        with plotutils.get_pyplot(inav=inav, name=cls.DESCR.upper()) as plt:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_axes([0.13, 0.1, 0.8, 0.8])
            line_style = '--' if any([sidx, eidx]) else '-'
            ax.plot(data.index, data.iloc[:, 0], line_style, label='average')
            if data.shape[-1] == 2 and data.iloc[:, 1].any():
                # Data has non-zero standard deviation column
                vals, errors = data.iloc[:, 0], data.iloc[:, 1]
                ax.fill_between(data.index,
                                vals - errors,
                                vals + errors,
                                color='y',
                                label='stdev',
                                alpha=0.3)
                ax.legend()
            if any([sidx, eidx]):
                gdata = data.iloc[sidx:eidx]
                ax.plot(gdata.index, gdata.iloc[:, 0], '.-g')
            xlabel = data.index.name
            # f"{cls.TIME_RE} ({starting_frame_index})" vs f"{regular_label}"
            xlabel = cls.TIME_LB if re.findall(cls.TIME_RE, xlabel) else xlabel
            ax.set_xlabel(xlabel)
            ax.set_ylabel(data.columns.values.tolist()[0])
            fname = f"{name}_{cls.NAME}{cls.FIG_EXT}"
            fig.savefig(fname)
            jobutils.add_outfile(fname)
        log(f'{cls.DESCR.capitalize()} figure saved as {fname}')

    @classmethod
    def read(cls, name, files=None, log=None, float_format='%.4g'):
        """
        Read the output file based on jobname or input files.

        :param name: the jobname based on which output file is searched
        :type name: str
        :param files: the output files from analyzers
        :type files: list of str
        :param log: the function to print user-facing information
        :type log: 'function'
        :param float_format str: the format to save float
        :return: x values, y average, y standard deviation
        :rtype: 'pandas.core.frame.DataFrame'
        """
        filename = f"{name}_{ cls.NAME}{cls.DATA_EXT}"
        if os.path.exists(filename):
            data = pd.read_csv(filename, index_col=0)
            log(f"{cls.RESULTS}{cls.DESCR} found as {filename}")
            return data
        if files is None:
            return
        datas = [pd.read_csv(x, index_col=0) for x in files]
        frm_num = min([x.shape[0] for x in datas])
        iname = min(datas, key=lambda x: x.shape[0]).index.name
        datas = [x.iloc[-frm_num:] for x in datas]
        xvals = [x.index.to_numpy().reshape(-1, 1) for x in datas]
        xvals = np.concatenate(xvals, axis=1)
        x_ave = xvals.mean(axis=1).reshape(-1, 1)
        yvals = [x.iloc[:, 0].to_numpy().reshape(-1, 1) for x in datas]
        yvals = np.concatenate(yvals, axis=1)
        y_std = yvals.std(axis=1).reshape(-1, 1)
        y_mean = yvals.mean(axis=1).reshape(-1, 1)
        data = np.concatenate((x_ave, y_mean, y_std), axis=1)
        data = pd.DataFrame(data[:, 1:], index=data[:, 0])
        cname, num = datas[0].columns[0], len(datas)
        data.columns = [f'{cname} (num={num})', f'std (num={num})']
        data.index.name = iname
        data.to_csv(filename, float_format=float_format)
        log(f"{cls.RESULTS}{cls.DESCR} saved to {filename}")
        return data

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


class Density(Base):
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
        mass = self.df_reader.molecular_weight / constants.Avogadro
        mass_scaled = mass / (constants.angstrom / constants.centi)**3
        data = [mass_scaled / x.volume for x in self.frms]
        self.data = pd.DataFrame({self.LABEL: data}, index=self.time)
        self.data.index.name = f"{self.ILABEL} ({self.sidx})"


class RDF(Base):
    """
    The radial distribution function analyzer.
    """

    NAME = 'rdf'
    DESCR = 'radial distribution function'
    PNAME = 'g'
    UNIT = 'r'
    LABEL = f'{PNAME} ({UNIT})'
    ILABEL = f'r ({symbols.ANGSTROM})'
    DEFAULT_CUT = lammpsin.In.DEFAULT_CUT

    def setData(self, res=0.02, dcut=None):
        """
        Set the radial distribution function.

        :param res float: the rdf minimum step
        :param dcut float: the cutoff distance to look for neighbors. If None,
            all the neighbors are counted when the cell is not significantly
             larger than the LJ cutoff.
        """
        if len(self.gids) < 2:
            self.log_warning("RDF requires least two atoms selected.")
            self.data = pd.DataFrame(data={self.LABEL: []})
            return
        # self.data = pd.DataFrame(data={self.LABEL: rdf}, index=index)
        frms = self.frms[self.sidx:]
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
        index = pd.Index(data=mid, name=self.ILABEL)
        self.data = pd.DataFrame(data={self.LABEL: rdf}, index=index)

    @classmethod
    def fit(cls, data, log=None):
        """
        Smooth the rdf data and report peaks.

        :param data: distance vs count
        :type data: 'pandas.core.frame.DataFrame'
        :param log: the function to print user-facing information
        :type log: 'function'
        :return int, int: the start and end index for the selected data
        """
        if data.empty:
            return None, None
        raveled = np.ravel(data[data.columns[0]])
        smoothed = savgol_filter(raveled, window_length=31, polyorder=2)
        row = data.iloc[smoothed.argmax()]
        log(f'Peak position: {row.name}; peak value: {row.values[0]: .2f}')
        return None, None


class MSD(Base):
    """
    The mean squared displacement analyzer.
    """

    NAME = 'msd'
    DESCR = 'mean squared displacement'
    PNAME = NAME.upper()
    UNIT = f'{symbols.ANGSTROM}^2'
    LABEL = f'{PNAME} ({UNIT})'
    ILABEL = 'Tau (ps)'

    def setData(self):
        """
        Set the mean squared displacement and diffusion coefficient.
        """
        if not self.gids:
            self.log_warning("No atoms selected for MSD.")
            self.data = pd.DataFrame({self.LABEL: []})
            return
        gids = list(self.gids)
        masses = self.df_reader.masses.mass[self.df_reader.atoms.type_id]
        frms = self.frms[self.sidx:]
        msd, num = [0], len(frms)
        for idx in range(1, num):
            disp = [
                x.xyz[gids, :] - y.xyz[gids, :]
                for x, y in zip(frms[idx:], frms[:-idx])
            ]
            data = np.array([np.linalg.norm(x, axis=1) for x in disp])
            sdata = np.square(data)
            msd.append(np.average(sdata.mean(axis=0), weights=masses))
        ps_time = self.time[self.sidx:][:num]
        tau_idx = pd.Index(data=ps_time - ps_time[0], name=self.ILABEL)
        self.data = pd.DataFrame({self.LABEL: msd}, index=tau_idx)

    @classmethod
    def fit(cls, data, spct=0.1, epct=0.2, log=None):
        """
        Select and fit the mean squared displacement to calculate the diffusion
        coefficient.

        :param data 'pandas.core.frame.DataFrame': time vs msd
        :param spct float: exclude the frames of this percentage at head
        :param epct float: exclude the frames of this percentage at tail
        :param log 'function': the function to print user-facing information
        :return int, int: the start and end index for the selected data
        """
        if data.empty:
            return None, None
        num = data.shape[0]
        sidx = math.floor(num * spct)
        eidx = math.ceil(num * (1 - epct))
        sel = data.iloc[sidx:eidx]
        # Standard error of the slope, under the assumption of residual normality
        xvals = sel.index * constants.pico
        yvals = sel.iloc[:, 0] * (constants.angstrom / constants.centi)**2
        slope, intercept, rvalue, p_value, std_err = linregress(xvals, yvals)
        # MSD=2nDt https://en.wikipedia.org/wiki/Mean_squared_displacement
        log(f'{slope/6:.4g} {symbols.PLUS_MIN} {std_err/6:.4g} cm^2/s'
            f' (R-squared: {rvalue**2:.4f}) linear fit of'
            f' [{sel.index.values[0]:.4f} {sel.index.values[-1]:.4f}] ps')
        return sidx, eidx


class Clash(Base):
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
        self.data.index.name = f"{self.ILABEL} ({self.sidx})"


class XYZ(Base):
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
        jobutils.add_outfile(self.outfile, jobname=self.options.jobname)
        self.log(f"{self.DESCR} coordinates are written into {self.outfile}")


class View(Base):
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
        jobutils.add_outfile(self.outfile, jobname=self.options.jobname)
        self.log(f'{self.DESCR.capitalize()} data written into {self.outfile}')


class Thermo(Base):

    NAME = 'thermo'
    DESCR = 'Thermodynamic information'

    @classmethod
    def plot(cls, data, name, *args, **kwargs):
        """
        Plot and save the data (interactively).

        :param data: data to plot
        :type data: 'pandas.core.frame.DataFrame'
        :param name: the taskname based on which output file is set
        :type name: str
        """
        for column in data.columns:
            aname = f"{name}_{column.split('(')[0].strip().lower()}"
            super().plot(data[column].to_frame(), aname, *args, **kwargs)
