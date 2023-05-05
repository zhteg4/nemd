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
                        f'{CLASH} check clashes for each frame.')

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
    DENSITY_EXT = f'_{DENSITY}.txt'
    MSD_EXT = f'_{MSD}.txt'
    RDF_EXT = f'_{RDF}.txt'
    RDF_PNG = f'_{RDF}.png'

    def __init__(self, options, jobname, diffusion=False):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param jobname str: jobname of this task
        :param diffusion bool: particles passing PBCs continue traveling
        """
        self.options = options
        self.jobname = jobname
        self.diffusion = diffusion
        self.outfile = self.jobname + self.XYZ_EXT
        self.out_density = self.jobname + self.DENSITY_EXT
        self.out_msd = self.jobname + self.MSD_EXT
        self.out_rdf = self.jobname + self.RDF_EXT
        self.rdf_png = self.jobname + self.RDF_PNG
        self.data_reader = None
        self.radii = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.checkClashes()
        self.writeXYZ()
        self.view()
        self.density()
        self.msd()
        self.rdf()
        log('Finished', timestamp=True)

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

    def checkClashes(self):
        """
        Check clashes for reach frames.
        """
        if CLASH not in self.options.task:
            return

        for idx, frm in enumerate(traj.Frame.read(self.options.custom_dump)):
            clashes = self.getClashes(frm)
            log(f"Frame {idx} has {len(clashes)} clashes.")
        log('All frames are checked for clashes.')

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
        frms = traj.get_frames(self.options.custom_dump)
        frm_vw.setFrames(frms)
        frm_vw.updateLayout()
        frm_vw.show()

    def density(self, last_pct=0.2, pname='Density', unit='g/cm^3'):
        """
        Calculate the density of all frames.

        :param last_pct float: average and std for the last frames of this percentage.
        :param pname str: property name
        :param unit str: unit of the property
        """

        if DENSITY not in self.options.task:
            return

        mass = self.data_reader.molecular_weight / constants.Avogadro
        mass_scaled = mass / (constants.angstrom / constants.centi)**3
        frms = traj.get_frames(self.options.custom_dump)
        data = [mass_scaled / x.getVolume() for x in frms]
        label = f'{pname} {unit}'
        data = pd.DataFrame({label: data})
        data.to_csv(self.out_density, float_format='%.4f')
        sfrm = math.floor(data.shape[0] * (1 - last_pct))
        sel = data.loc[sfrm:]
        ave = sel.mean()[label]
        std = sel.std()[label]
        msg = f'{ave:.4f} \u00B1 {std:.4f} {unit}  from frame {sfrm} to the end'
        log(msg)
        log(f'Density written into {self.out_density}')

    def msd(self, timestep=1, ex_pct=0.1, pname='MSD', unit='cm^2'):
        """
        Calculate the mean squared displacement and diffusion coefficient.

        :param timestep float: the time step in fs
        :param ex_pct float: fit the frames of this percentage at head and tail
        :param pname str: property name
        :param unit str: unit of the property
        """

        if MSD not in self.options.task:
            return
        masses = [
            self.data_reader.masses[x.type_id].mass
            for x in self.data_reader.atom
        ]
        frms = [x for x in traj.get_frames(self.options.custom_dump)]
        num = len(frms)
        msd = [0]
        for index in range(1, num):
            disp = [x - y for x, y in zip(frms[index:], frms[:-index])]
            data = np.array([np.linalg.norm(x, axis=1) for x in disp])
            sdata = np.square(data)
            msd.append(np.average(sdata.mean(axis=0), weights=masses))
        label = f'{pname} ({unit})'
        times = [x.getStep() for x in frms]
        data = pd.DataFrame({label: msd}, index=times)
        data[label] *= (constants.angstrom / constants.centi)**2
        data.index *= constants.femto * timestep
        data.index.name = 'Time (s)'
        sidx, eidx = math.floor(num * ex_pct), math.ceil(num * (1 - ex_pct))
        sel = data.iloc[sidx:eidx]
        slope, intercept, rvalue, p_value, std_err = scipy.stats.linregress(
            sel.index, sel[label])
        msg = f'{slope/6:.4g} \u00B1 {std_err:.4g} {unit}' \
              f' (R-squared: {rvalue**2:.4f})' \
              f' from frame {sidx} to {eidx}'
        log(msg)
        data.to_csv(self.out_msd, float_format='%.4g')
        log(f'Density written into {self.out_msd}')

    def rdf(self,
            last_pct=0.2,
            pname='g',
            unit='r',
            resolution=0.1,
            elements='O'):
        """
        Calculate the radial distribution function.

        :param last_pct float: the last frames of this percentage.
        :param pname str: property name
        :param unit str: unit of the property
        :param resolution float: the rdf minimum step
        :param elements str, set or list: the elements for atoms selection
        """

        if RDF not in self.options.task:
            return

        frms = [x for x in traj.get_frames(self.options.custom_dump)]
        sel = frms[math.floor(len(frms) * (1 - last_pct)):]
        span = np.array([[x for x in x.getSpan().values()] for x in sel])
        mdist = span.min() * 0.5
        bins = math.ceil(max([mdist / resolution, 20]))
        samples, bstep = np.linspace(mdist / bins,
                                     mdist,
                                     num=bins - 1,
                                     endpoint=False,
                                     retstep=True)
        mid = samples + bstep * 0.5
        ids = None
        if elements is not None:
            ids = [x.id for x in self.data_reader.atom if x.ele in elements]
        rdf = []
        for frm in sel:
            dists = frm.pairDists(ids=ids)
            hist, edges = np.histogram(dists,
                                       range=[bstep, mdist],
                                       bins=bins - 1)
            norm_factor = frm.getDensity() * 4 * np.pi * np.square(mid) * bstep
            rdf.append(hist * frm.shape[0] / sum(hist) / norm_factor)
        rdf = np.array(rdf).mean(axis=0)
        label = f'{pname} ({unit})'
        data = pd.DataFrame({label: rdf}, index=mid)
        data.index.name = 'r (\u212B)'
        data.to_csv(self.out_rdf, float_format='%.4f')
        log(f'Radial distribution function written into {self.out_rdf}')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(mid, rdf)
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(label)
        ax.set_xlim([data[data > 0].iloc[0].name, data.iloc[-1].name])
        fig.savefig(self.rdf_png)
        log(f'Radial distribution function figure saved as {self.rdf_png}')


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
