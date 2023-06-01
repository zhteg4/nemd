# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This post molecular dynamics driver perform trajectory analysis.
"""
import re
import sh
import os
import sys
import math
import scipy
import functools
import numpy as np
import pandas as pd
from scipy import constants
from scipy.signal import savgol_filter

from nemd import traj
from nemd import symbols
from nemd import oplsua
from nemd import jobutils
from nemd import logutils
from nemd import molview
from nemd import parserutils
from nemd import environutils

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE
FlAG_TASK = '-task'
FlAG_SEL = '-sel'
FLAG_LAST_PCT = '-last_pct'
FLAG_SLICE = '-slice'

CLASH = 'clash'
VIEW = 'view'
XYZ = 'xyz'
DENSITY = 'density'
MSD = 'msd'
RDF = 'rdf'

ALL_FRM_TASKS = [CLASH, VIEW, XYZ, DENSITY]
LAST_FRM_TASKS = [DENSITY, MSD, RDF]

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

ARGS_TMPL = [jobutils.FILE]


def log_debug(msg):
    """
    Print this message into the log file in debug mode.

    :param msg str: the msg to be printed
    """
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: print time after the msg
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


class CustomDump(object):
    """
    Analyze a dump custom file.
    """

    XYZ_EXT = '.xyz'
    DATA_EXT = '_%s.csv'
    PNG_EXT = '_%s.png'
    NAME = {
        XYZ: 'XYZ',
        CLASH: 'clash count',
        DENSITY: 'density',
        RDF: 'radial distribution function',
        MSD: 'mean squared displacement'
    }
    TIME_LB = 'Time (ps)'
    RESULTS = 'Results for '
    DEFAULT_CUT = oplsua.LammpsIn.DEFAULT_CUT

    def __init__(self, options, diffusion=False, timestep=1):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param diffusion bool: particles passing PBCs continue traveling
        :param timestep float: the time step in fs
        """
        self.options = options
        self.diffusion = diffusion
        self.timestep = timestep
        self.frms = None
        self.gids = None
        self.data_reader = None
        self.radii = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setAtoms()
        self.setFrames()
        self.checkClashes()
        self.view()
        self.density()
        self.msd()
        self.rdf()
        self.writeXYZ()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return
        self.data_reader = oplsua.DataFileReader(self.options.data_file)
        self.data_reader.run()
        if CLASH not in self.options.task:
            return
        self.data_reader.setClashParams()

    def setFrames(self):
        """
        Load trajectory frames and set range.
        """

        if self.options.slice is None:
            self.frms = [x for x in traj.get_frames(self.options.custom_dump)]
        else:
            frm_iter = traj.get_frames(self.options.custom_dump)
            [next(frm_iter) for _ in range(self.options.slice[0])]
            num = self.options.slice[1] - self.options.slice[0]
            frms = [next(frm_iter) for _ in range(num)]
            self.frms = frms[::self.options.slice[2]]
        self.time = np.array([x.getStep() for x in self.frms
                              ]) * constants.femto / constants.pico
        self.time_idx = pd.Index(data=self.time, name=self.TIME_LB)
        self.sidx = math.floor(len(self.frms) * (1 - self.options.last_pct))
        log(f"{len(self.frms)} trajectory frames found.")
        af_tasks = [x for x in self.options.task if x in ALL_FRM_TASKS]
        if af_tasks:
            log(f"{', '.join(af_tasks)} analyze all frames and save per frame "
                f"results {symbols.ELEMENT_OF} [{self.time_idx[0]:.3f}, "
                f"{self.time_idx[-1]:.3f}] ps")
        lf_tasks = [x for x in self.options.task if x in LAST_FRM_TASKS]
        if lf_tasks:
            log(f"{', '.join(lf_tasks)} average results from last "
                f"{self.options.last_pct * 100}% frames {symbols.ELEMENT_OF} "
                f"[{self.time_idx[self.sidx]: .3f}, {self.time_idx[-1]: .3f}] ps"
                )

    def checkClashes(self, label='Clash (num)'):
        """
        Check clashes for reach frames.
        """
        if CLASH not in self.options.task:
            return

        data = [len(self.getClashes(x)) for x in self.frms]
        data = pd.DataFrame(data={label: data}, index=self.time_idx)
        self.saveData(data, CLASH, float_format='%i')

    def getClashes(self, frm):
        """
        Get the clashes between atom pair for this frame.

        :param frm 'traj.Frame': traj frame to analyze clashes
        :return list of tuples: each tuple has two atom ids, the distance, and
            clash threshold
        """
        clashes = []
        dcell = traj.DistanceCell(frm=frm)
        dcell.setUp()
        for _, row in frm.iterrows():
            clashes += dcell.getClashes(row,
                                        radii=self.data_reader.radii,
                                        excluded=self.data_reader.excluded)
        return clashes

    def writeXYZ(self, wrapped=True, broken_bonds=False, glue=False):
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

        if XYZ not in self.options.task:
            return

        with open(self.outfile, 'w') as self.out_fh:
            for frm in traj.Frame.read(self.options.custom_dump):
                if wrapped:
                    frm.wrapCoords(broken_bonds, dreader=self.data_reader)
                if glue:
                    frm.glue(dreader=self.data_reader)
                frm.write(self.out_fh, dreader=self.data_reader)
        log(f"{self.NAME[XYZ]} coordinates are written into {self.outfile}")

    def view(self):
        """
        View the atom coordinates.
        """

        if VIEW not in self.options.task:
            return

        frm_vw = molview.FrameView(data_reader=self.data_reader)
        frm_vw.setData()
        frm_vw.setEleSz()
        frm_vw.setScatters()
        frm_vw.setLines()
        frm_vw.setEdges()
        frm_vw.addTraces()
        frm_vw.setFrames(self.frms)
        frm_vw.updateLayout()
        frm_vw.show()

    def density(self):
        """
        Calculate the density of all frames.
        """

        if DENSITY not in self.options.task:
            return

        data = self.getDensity()
        self.saveData(data, DENSITY)
        self.plot(data, DENSITY)

    def getDensity(self, pname='Density', unit='g/cm^3'):
        """
        Get the density data.

        :param pname str: property name
        :param unit str: unit of the property
        return 'pandas.core.frame.DataFrame': time and density
        """
        mass = self.data_reader.molecular_weight / constants.Avogadro
        mass_scaled = mass / (constants.angstrom / constants.centi)**3
        data = [mass_scaled / x.getVolume() for x in self.frms]
        label = f'{pname} {unit}'
        data = pd.DataFrame({label: data}, index=self.time)
        sel = data.loc[self.sidx:]
        ave = sel.mean()[label]
        std = sel.std()[label]
        log(f'{ave:.4f} {symbols.PLUS_MIN} {std:.4f} {unit} '
            f'{symbols.ELEMENT_OF} [{self.time[self.sidx]:.4f}, {self.time[-1]:.4f}] ps'
            )
        return data

    def msd(self):
        """
        Calculate the mean squared displacement and diffusion coefficient.

        :param ex_pct float: fit the frames of this percentage at head and tail
        :param pname str: property name
        :param unit str: unit of the property
        """
        # NOTE MSD needs frame selection
        if MSD not in self.options.task:
            return
        data = self.getMsd()
        self.saveData(data, MSD, float_format='%.4g')
        self.plot(data, MSD)

    def getMsd(self, ex_pct=0.1, pname='MSD', unit=f'{symbols.ANGSTROM}^2'):
        """
        Get the mean squared displacement and diffusion coefficient.

        :param ex_pct float: fit the frames of this percentage at head and tail
        :param pname str: property name
        :param unit str: unit of the property
        :return 'pandas.core.frame.DataFrame': time and msd
        """

        masses = [
            self.data_reader.masses[x.type_id].mass
            for x in self.data_reader.atom
        ]
        frms = self.frms[self.sidx:]
        msd, num = [0], len(frms)
        for idx in range(1, num):
            disp = [x - y for x, y in zip(frms[idx:], frms[:-idx])]
            data = np.array([np.linalg.norm(x, axis=1) for x in disp])
            sdata = np.square(data)
            msd.append(np.average(sdata.mean(axis=0), weights=masses))
        label = f'{pname} ({unit})'
        ps_time = self.time[self.sidx:][:num]
        tau_idx = pd.Index(data=ps_time - ps_time[0], name='Tau (ps)')
        data = pd.DataFrame({label: msd}, index=tau_idx)
        sidx, eidx = math.floor(num * ex_pct), math.ceil(num * (1 - ex_pct))
        sel = data.iloc[sidx:eidx]
        # Standard error of the slope, under the assumption of residual normality
        slope, intercept, rvalue, p_value, std_err = scipy.stats.linregress(
            sel.index * constants.pico,
            sel[label] * (constants.angstrom / constants.centi)**2)
        # MSD=2nDt https://en.wikipedia.org/wiki/Mean_squared_displacement
        log(f'{slope/6:.4g} {symbols.PLUS_MIN} {std_err/6:.4g} cm^2/s'
            f' (R-squared: {rvalue**2:.4f}) linear fit of'
            f' [{sel.index.values[0]:.4f} {sel.index.values[-1]:.4f}] ps '
            f'{symbols.ELEMENT_OF} [{self.time[self.sidx]:.4f}, {self.time[-1]:.4f}] ps'
            )
        return data

    def rdf(self):
        """
        Handle the radial distribution function.
        """

        if RDF not in self.options.task:
            return

        data = self.getRdf()
        self.saveData(data, RDF)
        self.plot(data, RDF, pos_y=True)

    def setAtoms(self):
        """
        set the atom selection for analysis.
        """
        if self.options.sel is None:
            self.gids = [x.id for x in self.data_reader.atom]
        else:
            self.gids = [
                x.id for x in self.data_reader.atom
                if x.ele in self.options.sel
            ]
        log(f"{len(self.gids)} atoms selected.")

    def getRdf(self, res=0.02, pname='g', unit='r', dcut=None):
        """
        Calculate and return the radial distribution function.

        :param res float: the rdf minimum step
        :param pname str: property name
        :param unit str: unit of the property
        :param dcut float: the cutoff distance to look for neighbors. If None,
            all the neighbors are counted when the cell is not significantly
             larger than the LJ cutoff.
        :return 'pandas.core.frame.DataFrame': pos and rdf
        """
        frms = self.frms[self.sidx:]
        span = np.array([[x for x in x.getSpan().values()] for x in frms])
        vol = np.prod(span, axis=1)
        log(f'The volume fluctuate: [{vol.min():.2f} {vol.max():.2f}] '
            f'{symbols.ANGSTROM}^3')
        # The auto resolution based on cut grabs left, middle, and right boxes
        if dcut is None and span.min() > self.DEFAULT_CUT * 5:
            # Cell is significant larger than LJ cut off, and thus use LJ cut
            dcut = self.DEFAULT_CUT
        if dcut:
            dres = dcut / 2
            # Grid the space up to 8000 boxes
            dres = span.min() / min([math.floor(span.min() / dres), 20])
            log(f"Only neighbors within {dcut} are accurate. (res={dres:.2f})")
        mdist = max(dcut, dres) if dcut else span.min() * 0.5
        res = min(res, mdist / 100)
        bins = round(mdist / res)
        hist_range = [res / 2, res * bins + res / 2]
        rdf, num = np.zeros((bins)), len(self.gids)
        for idx, frm in enumerate(frms):
            log_debug(f"Analyzing frame {idx} for RDF..")
            dists = frm.pairDists(ids=self.gids, cut=dcut, res=dres)
            hist, edge = np.histogram(dists, range=hist_range, bins=bins)
            mid = np.array([x for x in zip(edge[:-1], edge[1:])]).mean(axis=1)
            # 4pi*r^2*dr*rho from Radial distribution function - Wikipedia
            norm_factor = 4 * np.pi * mid**2 * res * num / frm.getVolume()
            # Stands at every id but either (1->2) or (2->1) is computed
            rdf += (hist * 2 / num / norm_factor)
        rdf /= len(frms)
        iname, cname = f'r ({symbols.ANGSTROM})', f'{pname} ({unit})'
        mid, rdf = np.concatenate(([0], mid)), np.concatenate(([0], rdf))
        index = pd.Index(data=mid, name=iname)
        data = pd.DataFrame(data={cname: rdf}, index=index)
        return data

    def saveData(self, data, task, float_format='%.4f'):
        """
        Save the data data.
        """
        outfile = self.options.jobname + self.DATA_EXT % task
        data.to_csv(outfile, float_format=float_format)
        log(f'{self.NAME[task].capitalize()} data written into {outfile}')

    def plot(self, data, task, pos_y=False):
        """
        Plot the task data and save the figure.

        :param data 'DataFrame': data to plot
        :param task str: the task type to get description and labels
        :param pos_y bool: change the xlim to only show data positive y values
        """
        fname = self.plotData(data,
                              task,
                              pos_y=pos_y,
                              inav=self.options.interactive,
                              sidx=self.sidx,
                              name=self.options.jobname)
        log(f'{self.NAME[task].capitalize()} figure saved as {fname}')

    @classmethod
    def plotData(cls,
                 data,
                 task,
                 pos_y=False,
                 inav=False,
                 sidx=None,
                 name=None):
        """
        :param data: data to plot
        :type data: 'pandas.core.frame.DataFrame'
        :param task: the task name
        :type task: str
        :param pos_y: set x lower limit to only include data with pos y value
        :type pos_y: bool
        :param inav: pop up window and show plot during code execution if
            interactive mode is on
        :type inav: bool
        :param sidx: the starting index when selecting data
        :type sidx: int
        :param name: the taskname based on which output file is set
        :type name: str
        :return: output file name
        :rtype: str
        """
        import matplotlib
        obackend = matplotlib.get_backend()
        backend = obackend if inav else 'Agg'
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(data.index, data.iloc[:, 0], label='average')
        if data.shape[-1] == 2:
            vals, errors = data.iloc[:, 0], data.iloc[:, 1]
            ax.fill_between(data.index,
                            vals - errors,
                            vals + errors,
                            color='y',
                            label='stdev')
            ax.legend()
        if sidx is not None and task in ALL_FRM_TASKS:
            ax.plot(data.index[sidx:], data.iloc[sidx:], 'g')
        if pos_y:
            ax.set_xlim([data[data > 0].iloc[0].name, data.iloc[-1].name])
        # ldata = list(data.values.flatten())
        # sidx = [x == ldata[0] for x in ldata].index(False)
        # ldata = list(reversed(ldata[sidx:]))
        # eidx = [x == ldata[0] for x in ldata].index(False)
        # frms = self.frms[sidx:-eidx]
        # frms = frms[round(len(frms) / 1.01*0.01):]
        # span = np.array([[y for y in x.getSpan().values()] for x in frms])
        # print(span.mean(axis=0))
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.columns.values.tolist()[0])
        fname = name + cls.PNG_EXT % task
        if inav:
            print(
                f"Showing {task}. Click X to close the figure and continue..")
            plt.show(block=True)
        fig.savefig(fname)
        matplotlib.use(obackend)
        return fname

    @classmethod
    def getOutfiles(cls, logfile):
        """
        Get the output files based on the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :return: task name and the related outfile
        :rtype: dict
        """
        jobname = cls.getLogged(logfile)[0]
        tasks = cls.getLoggedTasks(logfile)
        return {x: jobname + cls.DATA_EXT % x for x in tasks}

    @classmethod
    def getLoggedTasks(cls, logfile):
        """
        Get the task names in the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :return: task name
        :rtype: list
        """
        tasks = cls.getLogged(logfile, key=FlAG_TASK[1:])
        return [x.strip("'[]") for x in tasks]

    @classmethod
    def getLogged(cls, logfile, key=None):
        """
        Get the values corresponding to the key in the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :param key: the key based on which values are fetched
        :type key: str
        :return: the matching values in the logfile
        :rtype: list
        """
        if key is None:
            key = jobutils.FLAG_JOBNAME.lower()[1:]
        block = sh.grep(f'{key}:', logfile)
        return re.findall(f"(?<={key[1:]}: ).+(?=\n)", block)

    @classmethod
    def combine(cls, files, log, name, inav=False):
        """
        Combine multiple outfiles from the same task into one.

        :param files: task name and the related outfile
        :type files: dict
        :param log: the function to print user-facing information
        :type log: 'function'
        :param name: output files are named based on this name
        :type name: str
        :param inav: pop up window and show plot during code execution if
            interactive mode is on
        :type inav: bool
        """
        for tname, tfiles in files.items():
            if tname == XYZ:
                continue
            filename = f"{name}" + cls.DATA_EXT % tname
            dname = cls.NAME[tname]
            if os.path.exists(filename):
                log(f"{cls.RESULTS}{dname} found as {filename}")
                data = pd.read_csv(filename, index_col=0)
            else:
                datas = [pd.read_csv(x) for x in tfiles]
                frm_num = min([x.shape[0] for x in datas])
                datas = [x.iloc[-frm_num:] for x in datas]
                xvals = [x.iloc[:, 0].to_numpy().reshape(-1, 1) for x in datas]
                xvals = np.concatenate(xvals, axis=1)
                x_ave = xvals.mean(axis=1).reshape(-1, 1)
                yvals = [x.iloc[:, 1].to_numpy().reshape(-1, 1) for x in datas]
                yvals = np.concatenate(yvals, axis=1)
                y_std = yvals.std(axis=1).reshape(-1, 1)
                y_mean = yvals.mean(axis=1).reshape(-1, 1)
                data = np.concatenate((x_ave, y_mean, y_std), axis=1)
                data = pd.DataFrame(data[:, 1:], index=data[:, 0])
                cname, num = datas[0].columns[1], len(datas)
                data.columns = [f'{cname} (num={num})', f'std (num={num})']
                data.to_csv(filename)
                log(f"{cls.RESULTS}{dname} saved to {filename}")

            fname = cls.plotData(data, tname, name=name, inav=inav)
            log(f'{dname.capitalize()} figure saved as {fname}')
            if tname == RDF:
                raveled = np.ravel(data[data.columns[0]])
                smoothed = savgol_filter(raveled,
                                         window_length=31,
                                         polyorder=2)
                row = data.iloc[smoothed.argmax()]
                log(f'Peak position: {row.name}; '
                    f'peak value: {row.values[0]: .2f}')


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
        parser.add_argument(FlAG_CUSTOM_DUMP,
                            metavar=FlAG_CUSTOM_DUMP.upper(),
                            type=parserutils.type_file,
                            help='Custom dump file to analyze')
        parser.add_argument(FlAG_DATA_FILE,
                            metavar=FlAG_DATA_FILE[1:].upper(),
                            type=parserutils.type_file,
                            help='Data file to get force field information')
    parser.add_argument(FlAG_TASK,
                        choices=[XYZ, CLASH, VIEW, DENSITY, MSD, RDF],
                        default=[XYZ],
                        nargs='+',
                        help=f'{XYZ} writes out .xyz for VMD visualization;'
                        f' {CLASH} check clashes for each frame.')
    parser.add_argument(FlAG_SEL, help=f'Elements for atom selection.')
    parser.add_argument(
        FLAG_LAST_PCT,
        metavar=FLAG_LAST_PCT.upper(),
        type=functools.partial(parserutils.type_ranged_float,
                               include_top=False,
                               top=1),
        default=0.2,
        help=f"{', '.join(LAST_FRM_TASKS)} average results from "
        f"last frames of this percentage.")
    parser.add_argument(FLAG_SLICE,
                        metavar='START:END:INTERVAL',
                        type=parserutils.type_slice,
                        help=f"Slice the trajectory frames for analysis.")
    parserutils.add_job_arguments(parser,
                                  jobname=environutils.get_jobname(JOBNAME))
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)

    if CLASH in options.task and not options.data_file:
        parser.error(
            f'Please specify {FlAG_DATA_FILE} to run {FlAG_TASK} {CLASH}')
    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    cdump = CustomDump(options)
    cdump.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
