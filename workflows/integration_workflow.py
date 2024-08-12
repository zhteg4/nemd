# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This integration driver runs integration tests in one folder or all sub-folders.
The (sub-)folder name must be one integer to define the job id.
One test must contain one cmd file and a check file.

Supported check commands are: cmd, exist, not_exist ..
"""
import os
import sys
import glob
import pandas as pd

from nemd import logutils
from nemd import jobutils
from nemd import itestutils
from nemd import jobcontrol
from nemd import parserutils
from nemd import environutils

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_workflow', '')

FLAG_ID = 'id'
FLAG_DIR = itestutils.FLAG_DIR
FLAG_SLOW = '-slow'
FLAG_TASK = jobutils.FLAG_TASK
CMD = 'cmd'
CHECK = 'check'
TAG = 'tag'


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


class Integration(jobcontrol.Runner):
    """
    The main class to run integration tests.
    """

    def setJob(self):
        """
        Set operators. For example, operators to run cmd and check results.
        """
        if CMD in self.options.task:
            cmd = itestutils.CmdTask.getOpr(name='cmd')
        if CHECK in self.options.task:
            check = itestutils.CheckTask.getOpr(name='check')
            if CMD in self.options.task:
                self.setPrereq(check, cmd)
        if TAG in self.options.task:
            tag = itestutils.TagTask.getOpr(name='tag')
            if CMD in self.options.task:
                self.setPrereq(tag, cmd)

    def setState(self):
        """
        Set state with test dirs.
        """
        self.state = {FLAG_DIR: self.options.dir}

    def addJobs(self):
        """
        Add jobs to the project.
        """
        super().addJobs()
        if not self.options.clean:
            return
        for job in self.project.find_jobs():
            if CHECK in self.options.task:
                job.doc.pop(CHECK)
            if CHECK in self.options.task:
                job.doc.pop(CHECK)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(FLAG_ID,
                        metavar=FLAG_ID.upper(),
                        type=parserutils.type_positive_int,
                        nargs='+',
                        help='Select the sub-folders under the integration '
                        'test directory according to these ids.')
    parser.add_argument(FLAG_DIR,
                        metavar=FLAG_DIR[1:].upper(),
                        type=parserutils.type_dir,
                        default=environutils.get_integration_test_dir(),
                        help='The integration test directory.')
    parser.add_argument(
        FLAG_SLOW,
        type=parserutils.type_positive_float,
        metavar='SECOND',
        help='Skip tests marked with time longer than this criteria.')
    parser.add_argument(FLAG_TASK,
                        nargs='+',
                        choices=[CMD, CHECK, TAG],
                        default=[CMD, CHECK],
                        help='Select the tasks to run. cmd: run the cmd file; '
                        'check: check the results based on the check file;'
                        ' tag: update the tag file')
    parserutils.add_job_arguments(parser,
                                  jobname=environutils.get_jobname(JOBNAME))
    parserutils.add_workflow_arguments(
        parser, arg_flags=[parserutils.FLAG_CLEAN, parserutils.FLAG_JTYPE])
    return parser


def validate_options(argv):
    """
    Parse and validate the command options.

    :param argv list: command arguments
    :return 'argparse.Namespace': parsed command line options.
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    if not options.dir:
        parser.error(f'Please define the integration test dir via {FLAG_DIR}')

    if options.id:
        dirs = [os.path.join(options.dir, f"{x:0>4}") for x in options.id]
    else:
        dirs = glob.glob(os.path.join(options.dir, '[0-9]' * 4))

    options.dir = [x for x in dirs if os.path.isdir(x)]
    if not options.dir:
        parser.error(f'No valid tests found in {options.dir}.')

    if options.slow is not None:
        tags = [itestutils.TagParser(x, options=options) for x in options.dir]
        for tag in tags:
            tag.run()
        selected = [not x.slow for x in tags]
        options.dir = [x for x, y in zip(options.dir, selected) if y]
    if not options.dir:
        parser.error(f'All tests are marked as slow, skip running.')

    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    integration = Integration(options, argv, logger=logger)
    integration.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
