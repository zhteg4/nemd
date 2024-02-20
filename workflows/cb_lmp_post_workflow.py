# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This workflow driver runs crystal builder, lammps, and xxx.
"""
import os
import sys
import numpy as np
from flow import FlowProject

from nemd import logutils
from nemd import jobutils
from nemd import parserutils
from nemd import environutils
from nemd import jobcontrol
from nemd.task import Crystal_Builder, Lammps

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

FLAG_SCALED_RANGE = '-scaled_range'


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
    :param timestamp bool: the msg to be printed
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


@FlowProject.label
def label(job):
    """
    Show the label of a job (job id is a long alphanumeric string).

    :param job 'signac.contrib.job.Job': the job object
    :return str: the job name of a subtask
    """

    return str(job.statepoint())


class Runner(jobcontrol.Runner):

    CUSTOM_DUMP = 'custom_dump'

    def setTasks(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        polymer_builder = Crystal_Builder.getOpr(name='crystal_builder')
        lammps_runner = Lammps.getOpr(name='lammps_runner')
        self.setPrereq(lammps_runner, polymer_builder)
        # custom_dump = Custom_Dump.getOpr(name=self.CUSTOM_DUMP)
        # self.setPrereq(custom_dump, lammps_runner)

    def addJobs(self):
        """
        Add jobs to the project.
        """

        for scaled_range in np.arange(*self.options.scaled_range):
            job = self.project.open_job({self.STATE_ID: scaled_range})
            job.doc[jobutils.OUTFILE] = {}
            job.doc[jobutils.OUTFILES] = {}
            job.document[self.ARGS] = self.argv[:]
            job.document.update({self.PREREQ: self.prereq})
            job.init()

    # def setAggregation(self):
    #     """
    #     Aggregate post analysis jobs.
    #     """
    #     super().setAggregation()
    #     name = f"{self.jobname}{self.SEP}{self.CUSTOM_DUMP}"
    #     Custom_Dump.getAgg(name=name,
    #                        tname=self.CUSTOM_DUMP,
    #                        log=log,
    #                        clean=self.options.clean)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(
        FLAG_SCALED_RANGE,
        default=(0.9, 1.1, 0.05),  # yapf: disable
        nargs='+',
        metavar=FLAG_SCALED_RANGE.upper()[1:],
        type=parserutils.type_positive_float,
        help='Number of states for the dynamical system via random seed')
    parser = Crystal_Builder.DRIVER.get_parser(parser)

    parserutils.add_job_arguments(parser)
    parserutils.add_workflow_arguments(parser)
    return parser


def validate_options(argv):
    """
    Parse and validate the command options.

    :param argv list: command arguments
    :return 'argparse.Namespace': parsed command line options.
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


logger = None


def main(argv):
    global logger
    options = validate_options(argv)
    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    runner = Runner(options, argv, jobname, logger=logger)
    runner.run()
    log('finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
