# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This workflow driver runs polymer builder, lammps, and custom dump jobs.
"""
import functools
import os
import re
import sh
import sys
import pandas as pd
from flow import FlowProject, aggregator

from nemd import logutils
from nemd import jobutils
from nemd import parserutils
from nemd import environutils
from nemd import jobcontrol
from nemd.task import Polymer_Builder, Lammps, Custom_Dump

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

FLAG_STATE_NUM = '-state_num'


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

    def setTasks(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        polymer_builder = Polymer_Builder.getOpr(name='polymer_builder')
        lammps_runner = Lammps.getOpr(name='lammps_runner')
        self.setPrereq(lammps_runner, polymer_builder)
        custom_dump = Custom_Dump.getOpr(name='custom_dump')
        self.setPrereq(custom_dump, lammps_runner)

    def setAggregation(self):
        """
        Aggregate post analysis jobs.
        """
        Custom_Dump.getAgg(name=self.jobname, tname='custom_dump', log=log)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser = Polymer_Builder.DRIVER.get_parser(parser)
    parser = Custom_Dump.DRIVER.get_parser(parser)
    parser.add_argument(
        FLAG_STATE_NUM,
        default=1,
        metavar=FLAG_STATE_NUM.upper(),
        type=parserutils.type_positive_int,
        help='Number of states for the dynamical system via random seed')
    jobutils.add_job_arguments(parser)
    jobutils.add_workflow_arguments(parser)
    return parser


def validate_options(argv):
    """
    Parse and validate the command options.

    :param argv list: command arguments
    :return 'argparse.Namespace': parsed command line options.
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    if options.clean and jobutils.TASK not in options.jtype:
        parser.error(
            f'{jobutils.FLAG_CLEAN} removes the previous project, but '
            f'no new task jobs are found. ("{jobutils.FLAG_JTYPE} '
            f'{jobutils.TASK}" runs task jobs)')
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
