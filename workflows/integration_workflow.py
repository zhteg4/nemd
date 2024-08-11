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
import re
import sys
import glob
import datetime
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

    WORKSPACE = 'workspace'
    FLOW_PROJECT = 'flow.project'
    TAG = 'tag'
    SLOW = 'slow'
    TAG_KEYS = [SLOW]
    WILD_CARD = symbols.WILD_CARD
    IS_BASENAME_DIGIT_FUNC = lambda x: os.path.basename(x).isdigit()
    MSG = itestutils.ResultJob.MSG

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirs = None

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
        self.setTestDirs()
        self.skipTestDirs()
        self.state = {FLAG_DIR: self.dirs}

    def setTestDirs(self):
        """
        Set the test dirs by looking for the sub-folder tests or the input
        folder itself.
        """
        self.dirs = self.filterTestDir(self.options.dir)
        if not self.dirs:
            # Search for test dirs inside the input folder
            dirs = [os.path.join(x, self.WILD_CARD) for x in self.options.dir]
            pathnames = [y for x in dirs for y in glob.glob(x)]
            subdirs = filter(lambda x: os.path.isdir(x), pathnames)
            self.dirs = self.filterTestDir(subdirs)
        if not self.dirs:
            log_error(f'No tests found in {self.options.dir}.')
        log(f"{len(self.dirs)} tests found.")

    def filterTestDir(self, pathnames):
        """
        Filter the test directories from the given pathnames.

        :param pathnames list: the pathnames to filter.
        :return list: the filtered pathnames.
        """
        return list(filter(lambda x: os.path.basename(x).isdigit(), pathnames))

    def skipTestDirs(self):
        """
        Skip slow tests.
        """
        if self.options.slow is None:
            return
        orig_num = len(self.dirs)
        self.dirs = [x for x in self.dirs if not self.isSLow(x)]
        if not self.dirs:
            log_error(f'All tests in {self.options.dir} are skipped.')
        log(f"{orig_num - len(self.dirs)} tests skipped.")

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

    def addJobs(self):
        """
        Add jobs to the project.
        """
        super().addJobs()
        for job, test_dir in zip(self.project.find_jobs(), self.dirs):
            job.doc.pop(self.MSG)

    def logStatus(self):
        """
        Log message from the failed jobs in addition to the standard status log.
        """
        super().logStatus()
        jobs = self.project.find_jobs()
        fjobs = [x for x in jobs if x.document.get(self.MSG)]
        log(f"{len(jobs) - len(fjobs)} / {len(jobs)} succeeded jobs.")
        if not fjobs:
            return
        ids = [x.id for x in fjobs]
        msgs = [x.doc[self.MSG] for x in fjobs]
        dirs = [x.statepoint[FLAG_DIR] for x in fjobs]
        info = pd.DataFrame({'msg': msgs, 'dir': dirs}, index=ids)
        log(info.to_markdown())


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    itest_dir = environutils.get_integration_test_dir()
    parser.add_argument(FLAG_DIR,
                        metavar=FLAG_DIR.upper(),
                        type=parserutils.type_itest_dir,
                        default=[itest_dir] if itest_dir else None,
                        nargs='+',
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
