# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This workflow driver runs amorphous builder, lammps, and lmp_traj jobs.
"""
import os
import sys
from flow import FlowProject

from nemd import logutils
from nemd import jobutils
from nemd import parserutils
from nemd import jobcontrol
from nemd.task import AmorpBldr, Lammps, LmpTraj

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_workflow', '')

FLAG_STATE_NUM = jobutils.FLAG_STATE_NUM


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

    def setJob(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        amorphous_builder = self.setOpr(AmorpBldr, name='amorphous_builder')
        lammps_runner = self.setOpr(Lammps, name='lammps_runner')
        self.setPreAfter(amorphous_builder, lammps_runner)
        lmp_traj = self.setOpr(LmpTraj, name='lmp_traj')
        self.setPreAfter(lammps_runner, lmp_traj)

    def setAggJobs(self):
        """
        Aggregate post analysis jobs.
        """
        self.setAgg(LmpTraj, name='lmp_traj')
        super().setAggJobs()


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser = AmorpBldr.DRIVER.get_parser(parser)
    parser = LmpTraj.DRIVER.get_parser(parser)
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
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
    logger = logutils.createDriverLogger(jobname=JOBNAME, set_file=True)
    logutils.logOptions(logger, options)
    runner = Runner(options, argv, logger=logger)
    runner.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
