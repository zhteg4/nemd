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
import functools
import pandas as pd
from scipy import constants

from nemd import traj
from nemd import symbols
from nemd import lammpsdata
from nemd import jobutils
from nemd import logutils
from nemd import analyzer
from nemd import parserutils
from nemd import environutils

FLAG_CUSTOM_DUMP = parserutils.FLAG_CUSTOM_DUMP
FLAG_DATA_FILE = parserutils.FLAG_DATA_FILE
FLAG_TASK = jobutils.FLAG_TASK
FLAG_SEL = '-sel'
FLAG_LAST_PCT = '-last_pct'
FLAG_SLICES = '-slices'

CLASH = analyzer.Clash.NAME
VIEW = analyzer.View.NAME
XYZ = analyzer.XYZ.NAME
DENSITY = analyzer.Density.NAME
MSD = analyzer.MSD.NAME
RDF = analyzer.RDF.NAME

ALL_FRM_TASKS = [CLASH, VIEW, XYZ, DENSITY]
LAST_FRM_TASKS = [DENSITY, MSD, RDF]
DATA_RQD_TASKS = [CLASH, DENSITY, MSD, RDF]

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')
# Positional command-argument holders to take task input under jobcontrol
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
    TASK = FLAG_TASK[1:]
    DATA_EXT = '_%s.csv'
    PNG_EXT = '_%s.png'

    def __init__(self, options, timestep=1, unit=symbols.FS):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param timestep float: the time step
        :param unit str: the unit o time, default fs
        """
        self.options = options
        # FIXME: should read in timestep to calculate the time
        self.timestep = timestep
        self.unit = unit
        self.frms = None
        self.gids = None
        self.df_reader = None
        self.radii = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setAtoms()
        self.setFrames()
        self.analyze()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return
        self.df_reader = lammpsdata.DataFileReader(self.options.data_file)

    def setAtoms(self):
        """
        set the atom selection for analysis.
        """
        if not self.df_reader:
            return
        if self.options.sel is None:
            self.gids = self.df_reader.elements.index.tolist()
            log(f"{len(self.gids)} atoms selected.")
            return
        selected = self.df_reader.elements.element.isin([self.options.sel])
        self.gids = self.df_reader.elements.index[selected].tolist()
        log(f"{len(self.gids)} atoms selected.")

    def setFrames(self, start=0):
        """
        Load trajectory frames and set range.

        :param start int: only frames with step >= this value will be fully read
        """
        af_tasks = [x for x in self.options.task if x in ALL_FRM_TASKS]
        if not af_tasks:
            steps = traj.frame_steps(self.options.custom_dump)
            if len(steps) == 0:
                return
            sidx = math.floor(len(steps) * (1 - self.options.last_pct))
            if sidx == len(steps) - 1 and sidx:
                # From the second to the last frame in case of a broken last one
                sidx -= 1
            start = int(steps[sidx])

        frms = traj.slice_frames(self.options.custom_dump,
                                 slices=self.options.slices,
                                 start=start)
        self.frms = [x for x in frms]
        if len(self.frms) == 0:
            return

        self.sidx = math.floor(len(self.frms) * (1 - self.options.last_pct))
        self.time = pd.Index([x.step * self.timestep for x in self.frms])
        if self.unit == symbols.FS:
            self.time *= constants.femto / constants.pico
        self.time.name = symbols.TIME_ID.format(unit=symbols.PS, id=self.sidx)
        log(f"{len(self.frms)} trajectory frames found.")
        if af_tasks:
            log(f"{', '.join(af_tasks)} analyze all frames and save per frame "
                f"results {symbols.ELEMENT_OF} [{self.time[0]:.3f}, "
                f"{self.time[-1]:.3f}] ps")
        lf_tasks = [x for x in self.options.task if x in LAST_FRM_TASKS]
        if lf_tasks:
            log(f"{', '.join(lf_tasks)} average results from last "
                f"{self.options.last_pct * 100}% frames {symbols.ELEMENT_OF} "
                f"[{self.time[self.sidx]: .3f}, {self.time[-1]: .3f}] ps")

    def analyze(self):
        """
        Run analyzers.
        """
        if len(self.frms) == 0:
            log_error(f'{self.options.custom_dump} contains no frames.')
        for name in self.options.task:
            Analyzer = analyzer.ANALYZER[name]
            anl = Analyzer(self.time,
                           self.frms,
                           df_reader=self.df_reader,
                           gids=self.gids,
                           options=self.options,
                           logger=logger)
            anl.run()


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
        parser.add_argument(FLAG_CUSTOM_DUMP,
                            metavar=FLAG_CUSTOM_DUMP.upper(),
                            type=parserutils.type_file,
                            help='Custom dump file to analyze')
        parser.add_argument(FLAG_DATA_FILE,
                            metavar=FLAG_DATA_FILE[1:].upper(),
                            type=parserutils.type_file,
                            help='Data file to get force field information')
    parser.add_argument(FLAG_TASK,
                        choices=[XYZ, CLASH, VIEW, DENSITY, MSD, RDF],
                        default=[DENSITY],
                        nargs='+',
                        help=f'{XYZ} writes out .xyz for VMD visualization;'
                        f' {CLASH} checks clashes for each frame; {VIEW} '
                        f'visualizes coordinates; {DENSITY} analyzes the cell '
                        f'density; {MSD} computes mean squared displacement '
                        f'and diffusion coefficient; {RDF} calculates the '
                        f'radial distribution function. ')
    parser.add_argument(FLAG_SEL, help=f'The element of the selected atoms.')
    parser.add_argument(
        FLAG_LAST_PCT,
        metavar=FLAG_LAST_PCT.upper(),
        type=functools.partial(parserutils.type_ranged_float,
                               include_top=False,
                               top=1),
        default=0.2,
        help=f"{', '.join(LAST_FRM_TASKS)} average results from "
        f"last frames of this percentage.")
    parser.add_argument(FLAG_SLICES,
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
    data_rqd_tasks = set(options.task).intersection(DATA_RQD_TASKS)
    if data_rqd_tasks and not options.data_file:
        parser.error(f"Please specify {FLAG_DATA_FILE} to run {FLAG_TASK} "
                     f"{', '.join(data_rqd_tasks)}")

    try:
        options.task.remove(analyzer.XYZ.NAME)
    except ValueError:
        pass
    else:
        # XYZ analyzer may change the coordinates
        options.task.append(analyzer.XYZ.NAME)
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
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
