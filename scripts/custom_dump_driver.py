# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This post molecular dynamics driver perform trajectory analysis.
"""
import os
import sys
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
                        metavar=FlAG_DATA_FILE.upper(),
                        type=parserutils.type_file,
                        help='Data file to get force field information')
    parser.add_argument(FlAG_TASK,
                        choices=[XYZ, CLASH, VIEW],
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

        frm_vw = molview.FrameView(self.data_reader)
        frm_vw.setData()
        frm_vw.setScatters()
        frm_vw.setLines()
        frm_vw.addTraces()
        frms = traj.Frame.read(self.options.custom_dump)
        frm_vw.setFrames(frms)
        frm_vw.updateLayout()
        frm_vw.show()


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
