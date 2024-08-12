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

from nemd import symbols
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
FLAG_CHECK_ONLY = '-check_only'


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

    MSG = itestutils.CheckJob.MSG

    def setJob(self):
        """
        Set operators. For example, operators to run cmd and check results.
        """
        if self.options.check_only:
            itestutils.Result.getOpr(name='result')
            return

        cmd = itestutils.Cmd.getOpr(name='cmd')
        result = itestutils.Result.getOpr(name='result')
        self.setPrereq(result, cmd)

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
        for job in self.project.find_jobs():
            if self.options.check_only:
                job.doc.pop(self.MSG)

    def logStatus(self):
        """
        Log message from the failed jobs in addition to the standard status log.
        """
        super().logStatus()
        jobs = self.project.find_jobs()
        fjobs = [x for x in jobs if x.doc.get(self.MSG) is not False]
        log(f"{len(jobs) - len(fjobs)} / {len(jobs)} succeeded jobs.")
        if not fjobs:
            return
        ids = [x.id for x in fjobs]
        msgs = [x.doc.get(self.MSG, 'not run') for x in fjobs]
        dirs = [x.statepoint[FLAG_DIR] for x in fjobs]
        info = pd.DataFrame({'message': msgs, 'directory': dirs}, index=ids)
        log(info.to_markdown())


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
    parser.add_argument(FLAG_CHECK_ONLY,
                        action='store_true',
                        help='Checking for results only (skip the cmd task)')
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
        parser.error(f'No valid integration test dirs found in {options.dir}.')

    if options.slow is None:
        options.dir = [x for x in options.dir if not itestutils.Tag(x, options=options).isSlow()]
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
