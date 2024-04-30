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
import functools
import numpy as np
from scipy import constants

from nemd import traj
from nemd import symbols
from nemd import oplsua
from nemd import jobutils
from nemd import logutils
from nemd import analyzer
from nemd import parserutils
from nemd import environutils

FlAG_CUSTOM_DUMP = traj.FlAG_CUSTOM_DUMP
FlAG_DATA_FILE = traj.FlAG_DATA_FILE
FlAG_TASK = '-task'
FlAG_SEL = '-sel'
FLAG_LAST_PCT = '-last_pct'
FLAG_SLICE = '-slice'

CLASH = analyzer.Clash.NAME
VIEW = analyzer.View.NAME
XYZ = analyzer.XYZ.NAME
DENSITY = analyzer.Density.NAME
MSD = analyzer.MSD.NAME
RDF = analyzer.RDF.NAME

ALL_FRM_TASKS = [CLASH, VIEW, XYZ, DENSITY]
LAST_FRM_TASKS = [DENSITY, MSD, RDF]
DATA_RQD_TASKS = [CLASH, DENSITY, MSD, RDF]
NO_COMBINE = [XYZ, VIEW]

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
    TASK = FlAG_TASK[1:]
    DATA_EXT = '_%s.csv'
    PNG_EXT = '_%s.png'
    RESULTS = analyzer.BaseAnalyzer.RESULTS
    ANALYZER = [
        analyzer.Density, analyzer.RDF, analyzer.MSD, analyzer.Clash,
        analyzer.View, analyzer.XYZ
    ]
    ANALYZER = {getattr(x, 'NAME'): x for x in ANALYZER}

    def __init__(self, options, timestep=1):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param timestep float: the time step in fs
        """
        self.options = options
        # FIXME: should read in timestep to calculate the time
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
        self.analyze()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return
        self.data_reader = oplsua.DataFileReader(self.options.data_file)
        self.data_reader.run()

    def setAtoms(self):
        """
        set the atom selection for analysis.
        """
        if not self.data_reader:
            return
        if self.options.sel is None:
            self.gids = [x.id for x in self.data_reader.atom]
        else:
            self.gids = [
                x.id for x in self.data_reader.atom
                if x.ele in self.options.sel
            ]
        log(f"{len(self.gids)} atoms selected.")

    def setFrames(self):
        """
        Load trajectory frames and set range.
        """
        af_tasks = [x for x in self.options.task if x in ALL_FRM_TASKS]
        if af_tasks:
            frms = traj.slice_frames(self.options.custom_dump,
                                     slice=self.options.slice)
        else:
            steps = traj.frame_steps(self.options.custom_dump)
            sidx = math.floor(len(steps) * (1 - self.options.last_pct))
            frms = traj.slice_frames(self.options.custom_dump,
                                     slice=self.options.slice,
                                     start=int(steps[sidx]))
        self.frms = [x for x in frms]
        if len(self.frms) == 0:
            return
        self.time = np.array([x.step * self.timestep for x in self.frms
                              ]) * constants.femto / constants.pico
        self.sidx = math.floor(len(self.frms) * (1 - self.options.last_pct))
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
            Analyzer = self.ANALYZER[name]
            anl = Analyzer(self.time,
                           self.frms,
                           sidx=self.sidx,
                           data_reader=self.data_reader,
                           gids=self.gids,
                           options=self.options,
                           logger=logger)
            anl.run()

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
        tsks = cls.getLogged(logfile, key=cls.TASK, strip='[]', delimiter=', ')
        return {x: jobname + cls.DATA_EXT % x for x in tsks}

    @classmethod
    def getLogged(cls, logfile, key=None, strip=None, delimiter=None):
        """
        Get the values corresponding to the key in the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :param key: the key based on which values are fetched
        :type key: str
        :param delimiter: the chars to strip the string
        :type delimiter: str
        :param delimiter: the delimiter to split the string
        :type delimiter: str
        :return: the matching values in the logfile
        :rtype: list
        """
        if key is None:
            key = jobutils.FLAG_JOBNAME.lower()[1:]
        block = sh.grep(f'{key}:', logfile)
        matched = re.findall(f"(?<={key[1:]}: ).+(?=\n)", block)[0]
        matched = matched.strip(strip)
        matched = [x.strip() for x in matched.split(delimiter)]
        return matched

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
        for aname, afiles in files.items():
            if aname in NO_COMBINE:
                continue
            Analyzer = cls.ANALYZER[aname]
            data = Analyzer.read(name, files=afiles, log=log)
            sidx, eidx = Analyzer.fit(data, log=log)
            Analyzer.plot(data, name, inav=inav, sidx=sidx, eidx=eidx, log=log)


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
                        default=[DENSITY],
                        nargs='+',
                        help=f'{XYZ} writes out .xyz for VMD visualization;'
                        f' {CLASH} checks clashes for each frame; {VIEW} '
                        f'visualizes coordinates; {DENSITY} analyzes the cell '
                        f'density; {MSD} computes mean squared displacement '
                        f'and diffusion coefficient; {RDF} calculates the '
                        f'radial distribution function. ')
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
    data_rqd_tasks = set(options.task).intersection(DATA_RQD_TASKS)
    if data_rqd_tasks and not options.data_file:
        parser.error(f"Please specify {FlAG_DATA_FILE} to run {FlAG_TASK} "
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
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
