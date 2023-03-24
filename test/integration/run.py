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
import shutil
import sys
import glob
import filecmp
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import fileutils
from nemd import parserutils
from nemd import environutils
from nemd.nflow import FlowProject

ID = 'id'
DIR = 'dir'
MSG = 'msg'
SUCCESS = 'success'
CMD = 'cmd'
CHECK = 'check'
AND = 'and'
SIGNAC = 'signac'

JOBNAME = 'integration_test'

FLAG_DIR = DIR
FLAG_CLEAN = '-clean'


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
def cmd_completed(job):
    """
    The function to determine whether the main command has been executed.

    :param job 'signac.contrib.job.Job': the job object
    :return bool: whether the main command has been executed.

    NOTE：This should be modified when using with slurm schedular.
    """

    return bool([
        x for x in glob.glob(job.fn(symbols.WILD_CARD))
        if not os.path.basename(x).startswith(SIGNAC)
    ])


@FlowProject.post(cmd_completed)
@FlowProject.operation(cmd=True)
def run_cmd(job):
    """
    The method to run the main command.

    :param job 'signac.contrib.job.Job': the job object
    :return str: the shell command to execute
    """
    test_cmd_file = os.path.join(job.document[DIR], CMD)
    with open(test_cmd_file) as fh:
        lines = [x.strip() for x in fh.readlines()]
    cmd = symbols.SEMICOLON.join(lines)
    return f"cd {job.path}; {cmd}; cd -"


def checked(job):
    """
    The method to question whether the checking process has been performed.

    :param job 'signac.contrib.job.Job': the job object
    :return str: the shell command to execute
    """
    return SUCCESS in job.document


@FlowProject.pre.after(run_cmd)
@FlowProject.post(checked)
@FlowProject.operation
def check(job):
    """
    The method parses the check file and document the results.

    :param job 'signac.contrib.job.Job': the job object
    :return str: the shell command to execute
    """
    results = Results(job)
    try:
        results.run()
    except (FileNotFoundError, KeyError, ValueError) as err:
        job.document[SUCCESS] = False
        job.document[MSG] = str(err)
    else:
        job.document[SUCCESS] = True


class CMP:
    """
    The class to perform file comparison.
    """
    def __init__(self, original, target, job=None):
        self.orignal = original.strip().strip('\'"')
        self.target = target.strip().strip('\'"')
        self.job = job

    def run(self):
        """
        The main method to compare files.
        """
        self.orignal = os.path.join(self.job.document[DIR], self.orignal)
        if not os.path.isfile(self.orignal):
            raise FileNotFoundError(f"{self.orignal} not found")
        self.target = self.job.fn(self.target)
        if not os.path.isfile(self.target):
            raise FileNotFoundError(f"{self.target} not found")
        if not filecmp.cmp(self.orignal, self.target):
            raise ValueError(f"{self.target} is different from {self.orignal}")


class Results:
    """
    Class to parse the check file and execute the inside operations.
    """

    CMD_BRACKET_RE = '\s.*?\(.*?\)'
    PAIRED_BRACKET_RE = '\(.*?\)'
    CMD = {'cmp': CMP}

    def __init__(self, job):
        """
        :param job 'signac.contrib.job.Job': the signac job
        """
        self.job = job
        self.line = None
        self.operators = []

    def run(self):
        """
        Main method to get the results.
        """
        self.setLine()
        self.parserLine()
        self.executeOperators()

    def setLine(self):
        """
        Set the one line command by locating, reading, and cleaning the check file.
        """
        check_file = os.path.join(self.job.document[DIR], CHECK)
        with open(check_file) as fh:
            lines = [x.strip() for x in fh.readlines()]
        operators = [x for x in lines if not x.startswith(symbols.POUND)]
        self.line = ' ' + ' '.join(operators)

    def parserLine(self):
        """
        Parse the one line command to get the operators.
        """
        for operator in re.finditer(self.CMD_BRACKET_RE, self.line):
            operator = operator.group().strip()
            operator = operator.strip(AND + ' ').strip()
            self.operators.append(operator)

    def executeOperators(self):
        """
        Execute all operators. Raise errors during operation if one failed.
        """
        for operator in self.operators:
            self.execute(operator)

    def execute(self, operator):
        """
        Lookup the command class and execute.
        """
        bracketed = re.findall(self.PAIRED_BRACKET_RE, operator)[0]
        cmd = operator.replace(bracketed, '')
        try:
            runner_class = self.CMD[cmd]
        except KeyError:
            raise KeyError(
                f'{cmd} is one unknown command. Please select from {self.CMD.keys()}'
            )
        runner = runner_class(*bracketed[1:-1].split(','), job=self.job)
        runner.run()


class Integration:
    """
    The main class to run integration tests.
    """

    WORKSPACE = 'workspace'
    FLOW_PROJECT = 'flow.project'

    def __init__(self, options, jobname):
        """
        :param options 'argparse.Namespace': parsed commandline options.
        :param jobname str: the jobname
        """
        self.options = options
        self.jobname = jobname
        self.test_dirs = None
        self.project = None
        self.status_file = self.jobname + fileutils.STATUS_LOG
        # flow/project.py gets logger from logging.getLogger(__name__)
        logutils.createModuleLogger(self.FLOW_PROJECT, file_ext=fileutils.LOG)
        self.status_fh = None

    def run(self):
        """
        The main method to run the integration tests.
        """
        with open(self.status_file, 'w') as self.status_fh:
            self.clean()
            self.setTests()
            self.setProject()
            self.addJobs()
            self.runProject()
            self.logStatus()

    def clean(self):
        if not self.options.clean:
            return
        shutil.rmtree(self.WORKSPACE)

    def setTests(self):
        """
        Set the test dirs by looking for the sub-folder tests or the input
        folder itself.
        """
        base_dir = os.path.join(self.options.dir, symbols.WILD_CARD)
        self.test_dirs = [
            x for x in glob.glob(base_dir)
            if os.path.isdir(x) and os.path.basename(x).isdigit()
        ]
        if not self.test_dirs and os.path.basename(self.options.dir).isdigit():
            self.test_dirs = [self.options.dir]
        if not self.test_dirs:
            log_error(f'No tests found in {self.options.dir}.')
        log(f"{len(self.test_dirs)} tests found.")

    def setProject(self, workspace='workspace'):
        """
        Initiate the project.
        """
        self.project = FlowProject.init_project(workspace=workspace)

    def addJobs(self):
        """
        Add jobs to the project.
        """
        for test_dir in self.test_dirs:
            job = self.project.open_job({ID: os.path.basename(test_dir)})
            job.document[DIR] = test_dir
            job.init()

    def runProject(self):
        """
        Run all jobs registered in the project
        """
        self.project.run()

    def logStatus(self):
        """
        Look into each job and report the status.
        """
        # Fetching status and Fetching labels are printed to err handler
        self.project.print_status(detailed=True,
                                  file=self.status_fh,
                                  err=self.status_fh)

        jobs = self.project.find_jobs()
        sjobs = [x for x in jobs if x.document[SUCCESS]]
        fjobs = [x for x in jobs if not x.document[SUCCESS]]
        log(f"{len(sjobs)} succeed; {len(fjobs)} failed.")
        for fjob in fjobs:
            log(f'id: {fjob.sp[ID]}; dir: {os.path.basename(fjob.path)}')
            log(f' {fjob.document[MSG]}')
        log('finished.', timestamp=True)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(FLAG_DIR,
                        metavar=FLAG_DIR.upper(),
                        type=parserutils.type_dir,
                        nargs='?',
                        help='The directory to search for integration tests.')
    parser.add_argument(
        FLAG_CLEAN,
        action='store_true',
        help='Clean prevous results (if any) and run new ones.')
    jobutils.add_job_arguments(parser)
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
    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    integration = Integration(options, jobname)
    integration.run()


if __name__ == "__main__":
    main(sys.argv[1:])
