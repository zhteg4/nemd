# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This integration driver runs integration tests in one folder or all sub-folders.
The (sub-)folder name must be one integer to define the job id.
One test must contain one cmd file and a check file.

Supported check commands are:
cmd to compare two files;
..(more to come)
"""
import os
import re
import sys
import glob
import datetime
import argparse

from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import itestutils
from nemd import jobcontrol
from nemd import parserutils
from nemd import environutils

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

FLAG_DIR = itestutils.DIR
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

    WORKSPACE = 'workspace'
    FLOW_PROJECT = 'flow.project'
    TAG = 'tag'
    SLOW = 'slow'
    TAG_KEYS = [SLOW]

    def __init__(self, options, logger=None):
        """
        :param options 'argparse.Namespace': parsed commandline options.
        :param logger 'logging.Logger': print to this logger if exists.
        """
        super().__init__(options, [], logger=logger)
        self.test_dirs = None

    def setTasks(self):
        """
        Set integration tests as tasks.
        """
        self.setTests()
        self.skipTests()
        self.setOperators()

    def setTests(self):
        """
        Set the test dirs by looking for the sub-folder tests or the input
        folder itself.
        """
        self.test_dirs = [
            x for x in self.options.dir if os.path.basename(x).isdigit()
        ]
        if not self.test_dirs:
            self.test_dirs = [
                y for x in self.options.dir
                for y in glob.glob(os.path.join(x, symbols.WILD_CARD))
                if os.path.isdir(y) and os.path.basename(y).isdigit()
            ]
        if not self.test_dirs:
            log_error(f'No tests found in {self.options.dir}.')
        log(f"{len(self.test_dirs)} tests found.")

    def skipTests(self):
        """
        Skip slow tests.
        """
        if self.options.slow is None:
            return
        orig_num = len(self.test_dirs)
        self.test_dirs = [x for x in self.test_dirs if not self.isSLow(x)]
        if not self.test_dirs:
            log_error(f'All tests in {self.options.dir} are skipped.')
        log(f"{orig_num - len(self.test_dirs)} tests skipped.")

    def isSLow(self, test_dir):
        """
        Whether the test is slow and gets skipped.

        :param test_dir str: the directory of the test.
        :return bool: True when the test is marked with a time longer than the
            command line option requirement.
        """
        if self.options.slow is None:
            return False
        tags = self.getTags(test_dir, tag_keys=[self.SLOW])
        return tags.get(self.SLOW, 0) > self.options.slow

    def getTags(self, test_dir, tag_keys=None):
        """
        Get the tags in the test directory.

        :param test_dir str: the test directory.
        :param tag_keys list: tag keys to look for.
        :return dict: tag keys and values

        :raise ValueError: when the key is unknown.
        """
        if tag_keys is None:
            tag_keys = self.TAG_KEYS
        tag_file = os.path.join(test_dir, self.TAG)
        if not os.path.isfile(tag_file):
            return {}
        with open(tag_file) as tfh:
            lines = tfh.readlines()
        line = symbols.SEMICOLON.join(lines)
        tags = line.split(symbols.SEMICOLON)
        key_vals = {}
        for tag in tags:
            for tag_key in tag_keys:
                if tag.startswith(tag_key):
                    break
            else:
                raise ValueError(f"Unknown {tag} found in {tag_file}. Only "
                                 f"{self.TAG_KEYS} tags are supported.")
            if tag_key == self.SLOW:
                hms = re.search('\(.*?\)', tag).group()[2:-2]
                hms = datetime.datetime.strptime(hms, '%H:%M:%S')
                seconds = datetime.timedelta(
                    hours=hms.hour, minutes=hms.minute,
                    seconds=hms.second).total_seconds()
                key_vals[tag_key] = seconds
        return key_vals

    def setOperators(self):
        """
        Set operators. For example, run cmd and check results.
        """
        if self.options.check_only:
            itestutils.Results.getOpr(name='result', cmd=False)
            return
        cmd = itestutils.Integration_Driver.getOpr(name='cmd')
        result = itestutils.Results.getOpr(name='result', cmd=False)
        self.setPrereq(result, cmd)

    def addJobs(self):
        """
        Add jobs to the project.
        """

        for test_dir in self.test_dirs:
            statepoint = {self.STATE_ID: os.path.basename(test_dir)}
            job = self.project.open_job(statepoint)
            job.document[itestutils.DIR] = test_dir
            job.init()
            if self.options.check_only:
                job.doc.pop(itestutils.SUCCESS)

    def logStatus(self):
        """
        Log message from the failed jobs in addition to the standard status log.
        """
        super().logStatus()
        jobs = self.project.find_jobs()
        sjobs = [x for x in jobs if x.document[itestutils.SUCCESS]]
        fjobs = [x for x in jobs if not x.document[itestutils.SUCCESS]]
        log(f"{len(sjobs)} succeed; {len(fjobs)} failed.")
        for job in fjobs:
            if job.document[itestutils.SUCCESS]:
                continue
            id = job.statepoint[self.STATE_ID]
            dir = job.doc[itestutils.DIR]
            log(f'id: {id}; dir: {dir}')
            log(f'{job.document[itestutils.MSG]}')


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(FLAG_DIR,
                        metavar=FLAG_DIR.upper(),
                        type=parserutils.type_itest_dir,
                        nargs='?',
                        help='The directory to search for integration tests, '
                        f'or directories of the tests separated by '
                        f'\"{symbols.COMMA}\"')
    parser.add_argument(
        FLAG_SLOW,
        type=parserutils.type_positive_float,
        metavar='SECOND',
        help='Skip tests marked with time longer than this criteria.')
    parser.add_argument(FLAG_CHECK_ONLY,
                        action='store_true',
                        help='Checking for results only (the cmd task)')
    parserutils.add_job_arguments(parser,
                                  jobname=environutils.get_jobname(JOBNAME))
    parserutils.add_workflow_arguments(parser,
                                       arg_flags=[parserutils.FLAG_CLEAN])
    parser.add_argument(parserutils.FLAG_JTYPE,
                        default=[jobutils.TASK],
                        help=argparse.SUPPRESS)
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
        try:
            options.dir = environutils.get_integration_test_dir()
        except ValueError as err:
            parser.error(str(err))
    if isinstance(options.dir, str):
        options.dir = [options.dir]
    options.dir = [
        os.path.realpath(os.path.expanduser(x)) for x in options.dir
    ]
    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    integration = Integration(options, logger=logger)
    integration.run()
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
