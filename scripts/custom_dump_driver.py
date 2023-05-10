# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This post molecular dynamics driver perform trajectory analysis.
"""
import os
import sys
import math
import scipy
import numpy as np
import pandas as pd
from scipy import constants
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
CLASH = 'clash'
VIEW = 'view'
XYZ = 'xyz'
DENSITY = 'density'
MSD = 'msd'
RDF = 'rdf'

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


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


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
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

    jobutils.add_job_arguments(parser)
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


class CustomDump(object):
    """
    Analyze a dump custom file.
    """

    XYZ_EXT = '.xyz'
    DATA_EXT = '_%s.csv'
    PNG_EXT = '_%s.png'
    NAME = {
        CLASH: 'clash count',
        DENSITY: 'density',
        RDF: 'radial distribution function',
        MSD: 'mean squared displacement'
    }
    TIME_LB = 'Time (ps)'

    def __init__(self, options, jobname, diffusion=False, timestep=1):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param jobname str: jobname of this task
        :param diffusion bool: particles passing PBCs continue traveling
        :param timestep float: the time step in fs
        """
        self.options = options
        self.jobname = jobname
        self.diffusion = diffusion
        self.timestep = timestep
        self.outfile = self.jobname + self.XYZ_EXT
        self.frms = None
        self.data_reader = None
        self.radii = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setFrames()
        self.checkClashes()
        self.writeXYZ()
        self.view()
        self.density()
        self.msd()
        self.rdf()
        log('Finished.', timestamp=True)

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

    def setFrames(self, last_pct=0.2):
        """
        Load trajectory frames and set range.

        :param last_pct float: average and std for the last frames of this percentage.
        """

        self.frms = [x for x in traj.get_frames(self.options.custom_dump)
                     ]  #[:10]
        self.time = np.array([x.getStep() for x in self.frms
                              ]) * constants.femto / constants.pico
        self.time_idx = pd.Index(data=self.time, name=self.TIME_LB)
        self.sidx = math.floor(len(self.frms) * (1 - last_pct))

    def checkClashes(self, label='Clash (num)'):
        """
        Check clashes for reach frames.
        """
        if CLASH not in self.options.task:
            return

        data = [len(self.getClashes(frm)) for x in self.frms]
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
        log(f"Coordinates are written into {self.outfile}")

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
            f'{symbols.ELEMENT_OF} [{self.time[self.sidx]}, {self.time[-1]}] ps'
            )
        return data

    def msd(self, timestep=1, ex_pct=0.1, pname='MSD', unit='cm^2'):
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

    def rdf(self, elements='O'):
        """
        Calculate the radial distribution function.

        :param resolution float: the rdf minimum step
        :param elements str, set or list: the elements for atoms selection
        """

        if RDF not in self.options.task:
            return

        sel, ids = self.selectRdf(last_pct=0.2, elements=elements)
        data = self.getRdf(sel, ids)
        self.saveData(data, RDF)
        self.plot(data, RDF, pos_x=True)

    def selectRdf(self, last_pct=0.2, elements='O'):
        """
        Select the frame and element for rdf.

        :param elements str, set or list: the elements for atoms selection
        """
        frms = [x for x in traj.get_frames(self.options.custom_dump)]
        sel = frms[math.floor(len(frms) * (1 - last_pct)):]

        ids = [x.id for x in self.data_reader.atom if x.ele in elements
               ] if elements else [x.id for x in self.data_reader.atom]
        return sel, ids

    def getRdf(self, frms, ids, resolution=0.02, pname='g', unit='r'):
        """
        :param pname str: property name
        :param unit str: unit of the property
        return 'pandas.core.frame.DataFrame': pos and rdf
        """

        span = np.array([[x for x in x.getSpan().values()] for x in frms])
        mdist = span.min() * 0.5
        res = min(resolution, mdist / 100)
        bins = round(mdist / res)
        hist_range = [res / 2, res * bins + res / 2]
        rdf, num = [], len(ids)
        for frm in frms:
            dists = frm.pairDists(ids=ids)
            hist, edge = np.histogram(dists, range=hist_range, bins=bins)
            mid = np.array([x for x in zip(edge[:-1], edge[1:])]).mean(axis=1)
            # 4pi*r^2*dr*rho from Radial distribution function - Wikipedia
            norm_factor = 4 * np.pi * mid**2 * res * num / frm.getVolume()
            # Stands at every id but either (1->2) or (2->1) is computed
            rdf.append(hist * 2 / num / norm_factor)
        rdf = np.array(rdf).mean(axis=0)
        iname, cname = f'r ({symbols.ANGSTROM})', f'{pname} ({unit})'
        mid, rdf = np.concatenate(([0], mid)), np.concatenate(([0], rdf))
        index = pd.Index(data=mid, name=iname)
        return pd.DataFrame(data={cname: rdf}, index=index)

    def saveData(self, data, task, float_format='%.4f'):
        """
        Save the data data.
        """
        outfile = self.jobname + self.DATA_EXT % task
        data.to_csv(outfile, float_format=float_format)
        log(f'{self.NAME[task].capitalize()} data written into {outfile}')

    def plot(self, data, task, pos_x=False):
        """
        Plot the task data and save the figure.
        """
        import matplotlib
        obackend = matplotlib.get_backend()
        backend = obackend if self.options.interactive else 'Agg'
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(data.index, data)
        if pos_x:
            ax.set_xlim([data[data > 0].iloc[0].name, data.iloc[-1].name])
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.columns.values.tolist()[0])
        fname = self.jobname + self.PNG_EXT % task
        if self.options.interactive:
            print(
                f"Showing {task}. Click X to close the figure and continue..")
            plt.show(block=True)
        fig.savefig(fname)
        matplotlib.use(obackend)
        log(f'{self.NAME[task].capitalize()} figure saved as {fname}')


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    cdump = CustomDump(options, jobname)
    cdump.run()


if __name__ == "__main__":
    main(sys.argv[1:])
