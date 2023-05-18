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
import sh
import sys
import shutil
import datetime
import ordered_set
import numpy as np
import flow
from flow import FlowProject
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import fileutils
from nemd import parserutils
from nemd import environutils
import polymer_builder_driver
import custom_dump_driver

ID = 'id'
DIR = 'dir'
MSG = 'msg'
SUCCESS = 'success'
CMD = 'cmd'
CHECK = 'check'
AND = 'and'
ARGS = 'args'
KNOWN_ARGS = 'known_args'
UNKNOWN_ARGS = 'unknown_args'
SIGNAC = 'signac'

JOBNAME = 'integration_test'

FLAG_DIR = DIR
FLAG_CLEAN = '-clean'
FLAG_STATE_NUM = '-state_num'
STATE_ID = 'state_id'
WORKFLOW_NAME = 'workflow_name'


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


class BaseJob:

    NAME = 'name'
    TASK_ID = 'task_id'
    RUN_NEMD = jobutils.RUN_NEMD
    DRIVER_LOG = logutils.DRIVER_LOG
    FINISHED = jobutils.FINISHED

    def __init__(self, job):
        self.job = job
        self.document = self.job.document
        self.run_driver = [self.RUN_NEMD, self.DRIVER.PATH]

    def run(self):
        self.setTaskId()
        self.setArgs()
        self.setName()
        self.setCmd()

    def setTaskId(self):
        task_id = self.document.get(self.TASK_ID, 0)
        self.document.update({self.TASK_ID: task_id + 1})

    def setArgs(self):
        parser = self.DRIVER.get_parser()
        _, unknown = parser.parse_known_args(self.document[ARGS])
        self.document[UNKNOWN_ARGS] = unknown
        args = ordered_set.OrderedSet(self.document[ARGS])
        self.document[KNOWN_ARGS] = list(args.difference(unknown))

    def setName(self):
        name = self.getName(self.job)
        jobutils.set_arg(self.document[KNOWN_ARGS], jobutils.FLAG_JOBNAME, name)

    @classmethod
    def getName(cls, job):
        task_id = job.document.get(cls.TASK_ID, 1)
        return '_'.join([cls.DRIVER.JOBNAME, str(task_id)])

    def setCmd(self):
        cmd = self.run_driver + self.document[KNOWN_ARGS][:]
        import pdb;pdb.set_trace()
        self.cmd = [str(x) for x in cmd]

    def getCmd(self):
        return ' '.join(self.cmd)

    @classmethod
    def getLogfile(cls, job):
        name = cls.getName(job)
        return name + cls.DRIVER_LOG

    @classmethod
    def success(cls, job):
        logfile = job.fn(cls.getLogfile(job))
        return os.path.exists(logfile) and sh.tail('-2', logfile).startswith(
            cls.FINISHED)

    @classmethod
    def pre(cls, job):
        # import pdb; pdb.set_trace()
        # task_id = job.statepoint().get(cls.TASK_ID, 0)
        # job.update_statepoint({cls.TASK_ID: task_id + 1})
        return True

    # @classmethod
    # def post(cls, job):
    #     outfiles = job.document.pop(jobutils.OUTFILE, False)
    #     if outfiles:
    #         task_id = job.document.get('task_id', str(job.document['state_id']))
    #         task_id = [int(x) for x in task_id.split('_')]
    #         task_id = '_'.join(map(str, task_id[:-1] + [task_id[-1]+1]))
    #         # job.update_statepoint({'task_id': task_id})
    #         job.document[task_id] = outfiles
    #         job.document['task_id'] = task_id
    #     return outfiles


class Polymer_Builder(BaseJob):

    DRIVER = polymer_builder_driver
    FLAG_SEED = jobutils.FLAG_SEED

    # def __init__(self, job):
    #     super().__init__(job)

    def run(self):
        super().run()
        self.setSeed()

    def setSeed(self):
        seed = jobutils.get_arg(self.document[KNOWN_ARGS], self.FLAG_SEED)
        seed = int(seed) + self.document[STATE_ID]
        jobutils.set_arg(self.document[KNOWN_ARGS], self.FLAG_SEED, seed)

    @classmethod
    def post(cls, job):
        return job.document.get(jobutils.OUTFILE, {}).get(cls.getName(job))


class LAMMPS_RUN(BaseJob):

    # def __init__(self, job):
    #     super().__init__(job)

    @classmethod
    def pre(cls, job):
        return True

    @classmethod
    def post(cls, job):
        import pdb;pdb.set_trace()

class CUSTOM_DUMP(BaseJob):

    # def __init__(self, job):
    #     super().__init__(job)

    @classmethod
    def pre(cls, job):
        return True

    @classmethod
    def post(cls, job):
        import pdb;pdb.set_trace()


@FlowProject.label
def label(job):
    """
    Show the label of a job (job id is a long alphanumeric string).

    :param job 'signac.contrib.job.Job': the job object
    :return str: the job name of a subtask
    """

    return str(job.statepoint())


# @FlowProject.pre(lambda x: Polymer_Builder.pre(x))
@FlowProject.post(lambda x: Polymer_Builder.post(x))
@FlowProject.operation(cmd=True, with_job=True)
def polymer_builder(job):
    """
    Build cell.

    :param job 'signac.contrib.job.Job': the job object
    :return str: the shell command to execute
    """
    polymer_builder = Polymer_Builder(job)
    polymer_builder.run()
    return polymer_builder.getCmd()


# @FlowProject.pre.after(polymer_builder)
# @FlowProject.post(lambda x: LAMMPS_RUN.post(x))
# @FlowProject.operation(cmd=True, with_job=True)
# def simulation(job):
#     """
#     Run molecular dynamics simulation.
#
#     :param job 'signac.contrib.job.Job': the job object
#     :return str: the shell command to execute
#     """
#     import pdb;pdb.set_trace()
#     lammps_runner = LAMMPS_RUN(job)
#     lammps_runner.run()
#     return lammps_runner.getCmd()


# def checked(job):
#     """
#     The method to question whether the checking process has been performed.
#
#     :param job 'signac.contrib.job.Job': the job object
#     :return str: the shell command to execute
#     """
#     return SUCCESS in job.document

# @FlowProject.pre.after(polymer_builder)
# @FlowProject.post(checked)
# @FlowProject.operation
# def check(job):
#     """
#     The method parses the check file and document the results.
#
#     :param job 'signac.contrib.job.Job': the job object
#     :return str: the shell command to execute
#     """
#     results = Results(job)
#     try:
#         results.run()
#     except (FileNotFoundError, KeyError, ValueError) as err:
#         job.document[SUCCESS] = False
#         job.document[MSG] = str(err)
#     else:
#         job.document[SUCCESS] = True


class Runner:
    """
    The main class to run integration tests.
    """

    WORKSPACE = 'workspace'
    FLOW_PROJECT = 'flow.project'
    TAG = 'tag'
    SLOW = 'slow'
    TAG_KEYS = [SLOW]

    def __init__(self, options, argv, jobname):
        """
        :param options 'argparse.Namespace': parsed commandline options.
        :param jobname str: the jobname
        """
        self.options = options
        self.argv = argv
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
            self.setProject()
            self.addJobs()
            self.runProject()
            self.logStatus()

    def clean(self):
        """
        Remove the previous results on request.
        """
        if not self.options.clean:
            return
        try:
            shutil.rmtree(self.WORKSPACE)
        except FileNotFoundError:
            pass

    def skipTests(self):
        """
        Skip slow tests.
        """
        if self.options.slow is None:
            return
        orig_num = len(self.test_dirs)
        self.test_dirs = [x for x in self.test_dirs if not self.isSLow(x)]
        log(f"{orig_num - len(self.test_dirs)} tests skipped.")

    def setProject(self):
        """
        Initiate the project.
        """
        self.project = FlowProject.init_project()

    def addJobs(self):
        """
        Add jobs to the project.
        """
        for id in range(self.options.state_num):
            job = self.project.open_job({ID: id})
            job.document[ARGS] = self.argv[:]
            job.document[WORKFLOW_NAME] = self.jobname
            job.document[STATE_ID] = id
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
        status = [self.project.get_job_status(x) for x in jobs]
        completed = [all([y['completed'] for y in x['operations'].values()]) for x in status]
        [all([y['completed'] for y in x['operations'].values()]) for x in
         status]
        import pdb; pdb.set_trace()
        log(f"{len(completed)} / {len(status)} completed.")


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser = polymer_builder_driver.get_parser(parser)
    parser = custom_dump_driver.get_parser(parser)
    parser.add_argument(
        FLAG_CLEAN,
        action='store_true',
        help='Clean previous results (if any) and run new ones.')
    parser.add_argument(
        FLAG_STATE_NUM,
        default=1,
        metavar=FLAG_STATE_NUM.upper(),
        type=parserutils.type_positive_int,
        help='Number of states for the dynamical system via random seed')
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
    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    runner = Runner(options, argv, jobname)
    runner.run()
    log('finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
