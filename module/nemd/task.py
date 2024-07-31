import os
import re
import sh
import types
import logging
import argparse
import functools
import collections
import subprocess
import humanfriendly
import pandas as pd
from datetime import timedelta
from types import SimpleNamespace
from flow import FlowProject, aggregator

from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import lammpsin
from nemd import parserutils
from nemd import environutils

FILE = jobutils.FILE

logger = logutils.createModuleLogger(file_path=__file__)

SUCCESS = 'success'
MSG = 'msg'


def log_debug(msg):

    if logger is None:
        return
    logger.debug(msg)


class Job:

    TARGS = jobutils.TARGS
    SPECIAL_CHAR_RE = re.compile("[@!#$%^&*()<>?/|}{~:]")

    def __init__(self, job, name, driver):
        self.job = job
        self.name = name
        self.driver = driver
        self.doc = self.job.document
        self.args = None

    def setArgs(self):
        """
        Set arguments.
        """
        self.args = list(self.doc.get(self.TARGS, {}).get(self.name, []))
        self.args += list(self.doc[self.ARGS])
        self.args = list(map(str, self.args))

    def removeUnkArgs(self):
        """
        Set unknown arguments.

        Remove unknown arguments instead of keeping known so that the same flag
        across different tasks can be used multiple times.
        """
        parser = self.driver.get_parser()
        _, unknown = parser.parse_known_args(self.args)
        flags = [x for x in unknown if x.startswith('-')]
        # Positional arguments are ahead of optional arguments with flags
        positionals = unknown[:unknown.index(flags[0])] if flags else unknown
        # Remove unknown positional arguments without flags
        for arg in positionals:
            self.args.remove(arg)
        # Remove optional arguments with flags
        for flag, nfrag in zip(flags, flags[1:] + [None]):
            sidx = unknown.index(flag)
            eidx = unknown.index(nfrag) if nfrag else len(unknown)
            index = self.args.index(flag)
            self.args = self.args[:index] + self.args[index + eidx - sidx:]

    def setName(self):
        """
        Set the jobname of the known args.
        """
        jobutils.set_arg(self.args, jobutils.FLAG_JOBNAME, self.name)

    def addQuote(self):
        """
        Add quotes for str with special characters.
        """
        quote_needed = self.SPECIAL_CHAR_RE.search
        self.args = [f'"{x}"' if quote_needed(x) else x for x in self.args]


class BaseTask:
    """
    The task base class.
    """

    ID = 'id'
    TIME = 'time'
    STIME = 'stime'
    ETIME = 'etime'
    STATE_ID = jobutils.STATE_ID
    TIME_REPORTED = 'time_reported'
    TIME_BREAKDOWN = 'Task timing breakdown:'
    SEP = symbols.SEP
    ARGS = jobutils.ARGS
    TARGS = jobutils.TARGS
    PREREQ = jobutils.PREREQ
    OUTFILE = jobutils.OUTFILE
    RUN_NEMD = jobutils.RUN_NEMD
    FINISHED = jobutils.FINISHED
    DRIVER_LOG = logutils.DRIVER_LOG
    KNOWN_ARGS = jobutils.KNOWN_ARGS
    UNKNOWN_ARGS = jobutils.UNKNOWN_ARGS
    DRIVER = SimpleNamespace(PATH=None)
    QUOTED_CHAR = re.compile("[@!#$%^&*()<>?/|}{~:]")

    def __init__(self, job, pre_run=RUN_NEMD, name=None):
        """
        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param pre_run: append this str before the driver path
        :type pre_run: str
        :param name: the jobname
        :type name: str
        """
        self.job = job
        self.pre_run = pre_run
        self.name = name
        self.doc = job.document
        self.run_driver = [self.DRIVER.PATH]
        if self.pre_run:
            self.run_driver = [self.pre_run] + self.run_driver

    def run(self):
        """
        The main method to run.
        """
        self.setArgs()
        self.setName()
        self.addQuote()

    def setArgs(self):
        """
        Set known and unknown arguments.
        """
        parser = self.DRIVER.get_parser()
        args = list(self.doc.get(self.TARGS, {}).get(self.name, []))
        args += list(self.doc[self.ARGS])
        _, unknown = parser.parse_known_args(args)
        # self.doc[self.UNKNOWN_ARGS] = unknown
        flags = [x for x in unknown if x.startswith('-')]
        # Positional arguments without flags
        pos_unknown = unknown[:unknown.index(flags[0])] if flags else unknown
        for arg in pos_unknown:
            args.remove(arg)
        for sval, eval in zip(flags, flags[1:] + [None]):
            if eval:
                uargs = unknown[unknown.index(sval):unknown.index(eval)]
            else:
                uargs = unknown[unknown.index(sval):]
            index = args.index(sval)
            # Optional arguments with flags
            args = args[:index] + args[index + len(uargs):]
        self.doc[self.KNOWN_ARGS] = args

    def setName(self):
        """
        Set the jobname of the known args.
        """
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], jobutils.FLAG_JOBNAME,
                         self.name)

    def getCmd(self, write=True, extra_args=None):
        """
        Get command line str.

        :param write bool: the msg to be printed
        :param extra_args list: extra args for the specific task
        :return str: the command as str
        """
        # self.doc[KNOWN_ARGS] is not a list but BufferedJSONAttrLists
        args = list(self.doc[self.KNOWN_ARGS])
        if extra_args:
            args += extra_args
        cmd = ' '.join(list(map(str, self.run_driver + args)))
        if write:
            with open(f"{self.name}_cmd", 'w') as fh:
                fh.write(cmd)
        return cmd

    def addQuote(self):
        """
        Add quotes for str with special characters.
        """
        self.doc[self.KNOWN_ARGS] = [
            f'"{x}"' if self.QUOTED_CHAR.search(str(x)) else x
            for x in self.doc[self.KNOWN_ARGS]
        ]

    @classmethod
    def success(cls, job, name):
        """
        Whether job is successful based on job logging.

        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname
        :type name: str
        :return: whether job is successful
        :rtype: bool
        """
        logfile = job.fn(name + cls.DRIVER_LOG)
        return os.path.exists(logfile) and sh.tail('-2', logfile).split(
            symbols.RETURN)[0].endswith(cls.FINISHED)

    @classmethod
    def pre(cls, job, name=None):
        """
        Set and check pre-conditions before starting the job.

        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname
        :type name: str
        :return: True if the pre-conditions are met
        :rtype: bool
        """
        try:
            pre_jobs = job.doc[cls.PREREQ][name]
        except KeyError:
            # The job doesn't have any prerequisite jobs
            log_debug(f'Pre-conditions: {name} (True): no pre-conditions')
            return True
        if not all(job.doc[cls.OUTFILE][x] for x in pre_jobs):
            # The job has incomplete prerequisite jobs and has to wait
            log_debug(
                f'Pre-conditions: {name} (False): {[f"{x} ({job.doc[cls.OUTFILE][x]})" for x in pre_jobs]}'
            )
            return False
        args = cls.DRIVER.ARGS_TMPL[:]
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for pre in job.doc[cls.PREREQ][name]:
            index = args.index(FILE)
            args[index] = job.doc[cls.OUTFILE][pre]
        job.doc.setdefault(cls.TARGS, {})[name] = args
        log_debug(f'Pre-conditions: {name} (True): {args}')
        return True

    @classmethod
    def post(cls, job, name=None):
        """
        Check post-conditions after the job has started (FIXME: I am not sure
        whether this runs during the execution. I guess that this runs after
        the execution but before the job is considered as completion. Please
        revise or correct the doc string after verification).

        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname
        :type name: str
        :return: True if the post-conditions are met
        :rtype: bool
        """
        outfile = job.document.get(cls.OUTFILE, {}).get(name)
        log_debug(f'Post-conditions: {name}: {outfile}')
        if outfile:
            return True
        return False

    @classmethod
    def operator(cls, job, *arg, name=None, **kwargs):
        """
        The main opterator (function) for a job task executed after
        pre-conditions are met.
        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname
        :type name: str
        :return str: the command to run a task.
        """
        obj = cls(job, name=name)
        obj.run()
        return obj.getCmd()

    @classmethod
    def getOpr(cls,
               cmd=True,
               with_job=True,
               name=None,
               attr='operator',
               pre=None,
               post=None,
               aggregator=None,
               log=None,
               tname=None,
               **kwargs):
        """
        Duplicate and return the operator with jobname and decorators.

        :param cmd: Whether the aggregator function returns a command to run
        :type cmd: bool
        :param with_job: perform the execution in job dir with context management
        :type with_job: bool
        :param name: the taskname
        :type name: str
        :param attr: the attribute name of a staticmethod method or callable function
        :type attr: str or types.FunctionType
        :param pre: add pre-condition for the aggregator if True
        :type pre: bool
        :param post: add post-condition for the aggregator if True
        :type post: bool
        :param aggregator: the criteria to collect jobs
        :type aggregator: 'flow.aggregates.aggregator'
        :param log: the function to print user-facing information
        :type log: 'function'
        :param tname: aggregate the job tasks of this name
        :type tname: str
        :return: the operation to execute
        :rtype: 'function'
        :raise ValueError: the function cannot be found
        """
        if post is None:
            post = cls.post
        if pre is None:
            pre = cls.pre

        if isinstance(attr, str):
            opr = getattr(cls, attr)
        elif isinstance(attr, types.FunctionType):
            opr = attr
        else:
            raise ValueError(f"{attr} is not a callable function or str.")

        # Pass jobname, taskname, and logging function
        kwargs.update({'name': name})
        if tname:
            kwargs['tname'] = tname
        if log:
            kwargs['log'] = log
        func = functools.update_wrapper(functools.partial(opr, **kwargs), opr)
        func.__name__ = name

        func = FlowProject.operation(cmd=cmd,
                                     func=func,
                                     with_job=with_job,
                                     name=name,
                                     aggregator=aggregator)
        # Add FlowProject decorators (pre / post conditions)
        if post:
            fp_post = functools.partial(post, name=name)
            fp_post = functools.update_wrapper(fp_post, post)
            func = FlowProject.post(lambda *x: fp_post(*x))(func)
        if pre:
            fp_pre = functools.partial(pre, name=name)
            fp_pre = functools.update_wrapper(fp_pre, pre)
            func = FlowProject.pre(lambda *x: fp_pre(*x))(func)
        log_debug(f'Operator: {func.__name__}: {func}')
        return func

    @classmethod
    def aggregator(cls, *jobs, log=None, **kwargs):
        """
        The aggregator job task to report the time cost of each task.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param log: the function to print user-facing information
        :type log: 'function'
        """

        log(BaseTask.TIME_BREAKDOWN)
        info = collections.defaultdict(list)
        for job in jobs:
            for tname, filename in job.doc[jobutils.LOGFILE].items():
                delta = logutils.get_time(job.fn(filename))
                info[tname].append(SimpleNamespace(delta=delta, id=job.id))
        for tname, tinfo in info.items():
            tinfo = [x for x in tinfo if x.delta is not None]
            if not tinfo:
                continue
            total = sum([x.delta for x in tinfo], start=timedelta(0))
            ave = humanfriendly.format_timespan(total / len(tinfo))
            deltas = [
                f"{humanfriendly.format_timespan(x.delta)} ({x.id[:4]})"
                for x in tinfo
            ]
            log(f"{tname}: {', '.join(deltas)}; {ave} (ave)")

    @classmethod
    def getAgg(cls,
               cmd=False,
               with_job=False,
               name=None,
               attr='aggregator',
               pre=False,
               post=None,
               log=None,
               tname=None,
               **kwargs):
        """
        Get and register an aggregator job task that collects task outputs.

        :param cmd: Whether the aggregator function returns a command to run
        :type cmd: bool
        :param with_job: Whether chdir to the job dir
        :type with_job: bool
        :param name: the name of this aggregator job task.
        :type name: str
        :param attr: the attribute name of a staticmethod method or callable function
        :type attr: str or types.FunctionType
        :param pre: add pre-condition for the aggregator if True
        :type pre: bool
        :param post: add post-condition for the aggregator if True
        :type post: bool
        :param log: the function to print user-facing information
        :type log: 'function'
        :param tname: aggregate the job tasks of this name
        :type tname: str
        :return: the operation to execute
        :rtype: 'function'
        """
        if post is None:
            post = cls.postAgg
        return cls.getOpr(aggregator=aggregator(),
                          cmd=cmd,
                          with_job=with_job,
                          name=name,
                          attr=attr,
                          pre=pre,
                          post=post,
                          log=log,
                          tname=tname,
                          **kwargs)

    @classmethod
    def postAgg(cls, *jobs, name=None):
        """
        Post-condition for task time reporting.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param name: jobname based on which log file is found
        :type name: str
        :return: the label after job completion
        :rtype: str
        """
        log_file = name + logutils.DRIVER_LOG
        try:
            sh.grep(cls.TIME_BREAKDOWN, log_file)
        except sh.ErrorReturnCode_1:
            return False
        return cls.TIME_REPORTED

    @staticmethod
    def suppress(parser, to_supress=None):
        """
        Suppress certain command line arguments.

        :param parser 'argparse.ArgumentParser': the argument parser object
        :param to_supress list: the list of arguments to suppress
        """
        if to_supress is None:
            return
        to_supress = set(to_supress)
        for action in parser._actions:
            if to_supress.intersection(action.option_strings):
                action.help = argparse.SUPPRESS


class Polymer_Builder(BaseTask):

    import polymer_builder_driver as DRIVER


class Crystal_Builder(BaseTask):

    import crystal_builder_driver as DRIVER


class Lammps_Driver:
    """
    LAMMPS wrapper to package the imported software as a nemd driver.
    """

    LMP_SERIAL = 'lmp_serial'
    PATH = LMP_SERIAL
    JOBNAME = 'lammps'
    FLAG_IN = '-in'
    FLAG_SCREEN = '-screen'
    FLAG_LOG = '-log'
    DRIVER_LOG = '_lammps.log'

    ARGS_TMPL = [FLAG_IN, FILE, FLAG_SCREEN, 'none']

    @classmethod
    def get_parser(cls):
        """
        Get the customized parser wrapper for lammps executable.

        :return: the customized parser wrapper
        :rtype: 'argparse.ArgumentParser'
        """
        parser = parserutils.get_parser(
            description='This is a customized parser wrapper for lammps.')
        parser.add_argument(cls.FLAG_IN,
                            metavar='IN_SCRIPT',
                            type=parserutils.type_file,
                            required=True,
                            help='Read input from this file.')
        parser.add_argument(cls.FLAG_SCREEN,
                            choices=['none', 'filename'],
                            help='where to send screen output (-sc)')
        parser.add_argument(cls.FLAG_LOG,
                            help='Print logging information into this file.')
        return parser


class Lammps(BaseTask):

    DRIVER = Lammps_Driver

    def __init__(self, *args, pre_run=None, **kwargs):
        """
        :param pre_run: lammps driver itself is a executable file omitting pre_run
        :type pre_run: None or str
        """
        super().__init__(*args, pre_run=pre_run, **kwargs)

    @classmethod
    def operator(cls, *arg, **kwargs):
        """
        Get the lammps operation command.
        """
        cmd = super().operator(*arg, **kwargs)
        # run_nemd echo Running xxx
        return f'echo Running {cmd}; {cmd}'

    def run(self):
        """
        The main method to run.
        """
        super().run()
        self.setLammpsLog()
        self.setDriverLog()

    def setName(self):
        """
        Overwrite the parent as lammps executable doesn't take jobname flag.
        """
        pass

    def getCmd(self, write=True):
        """
        Get command line str.

        :param write bool: the msg to be printed
        :return str: the command as str
        """
        lmp = subprocess.run(f'{self.DRIVER.LMP_SERIAL} -h | grep GPU',
                             capture_output=True,
                             shell=True)
        extra_args = ['-sf', 'gpu', '-pk', 'gpu', '1'] if lmp.stdout else None
        return super().getCmd(write=write, extra_args=extra_args)

    def setLammpsLog(self):
        """
        Set the output log name based on jobname.
        """
        logfile = self.name + self.DRIVER.DRIVER_LOG
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], '-log', logfile)

    def setDriverLog(self):
        parser = self.DRIVER.get_parser()
        options = parser.parse_args(self.doc[self.KNOWN_ARGS])
        logger = logutils.createDriverLogger(jobname=self.name)
        logutils.logOptions(logger, options)
        logutils.log(logger, 'Running lammps simulations..')

    @classmethod
    def post(cls, job, name=None):
        """
        Set the output for the job.

        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname
        :type name: str
        :return: True if the post-conditions are met
        :rtype: bool
        """

        basename = name + cls.DRIVER.DRIVER_LOG
        logfile = job.fn(basename)
        if not os.path.exists(logfile):
            return False
        if super().post(job, name):
            return True
        if not os.popen(f'tail -2 {logfile}').read().startswith(
                'ERROR') and os.popen(f'tail -1 {logfile}').read().startswith(
                    'Total'):
            jobutils.add_outfile(basename,
                                 jobname=name,
                                 job=job,
                                 document=job.fn(jobutils.FN_DOCUMENT),
                                 set_file=True)
            logger = logging.getLogger(name)
            logutils.log(logger, logutils.FINISHED, timestamp=True)
        return super().post(job, name)


class Custom_Dump(BaseTask):

    import custom_dump_driver as DRIVER
    CUSTOM_EXT = lammpsin.In.CUSTOM_EXT
    DUMP = lammpsin.In.DUMP
    READ_DATA = lammpsin.In.READ_DATA
    DATA_EXT = lammpsin.In.DATA_EXT
    RESULTS = DRIVER.CustomDump.RESULTS

    def setArgs(self):
        """
        Set the args for custom dump task.
        """
        super().setArgs()
        log_file = self.doc[self.KNOWN_ARGS][0]
        dump_cmd = sh.grep(self.DUMP, log_file).split()
        dump_file = [x for x in dump_cmd if x.endswith(self.CUSTOM_EXT)][0]
        data_cmd = sh.grep(self.READ_DATA, log_file).split()
        data_file = [x for x in data_cmd if x.endswith(self.DATA_EXT)][0]
        args = [dump_file, self.DRIVER.FLAG_DATA_FILE, data_file]
        args += list(self.doc[self.KNOWN_ARGS])[1:]
        self.doc[self.KNOWN_ARGS] = args

    @classmethod
    def aggregator(cls, *jobs, log=None, name=None, tname=None, **kwargs):
        """
        The aggregator job task that combines the output files of a custom dump
        task.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param log: the function to print user-facing information
        :type log: 'function'
        :param name: the jobname based on which output files are named
        :type name: str
        :param tname: aggregate the job tasks of this name
        :type tname: str
        """
        log(f"{len(jobs)} jobs found for aggregation.")
        job = jobs[0]
        logfile = job.fn(job.document[jobutils.OUTFILE][tname])
        outfiles = cls.DRIVER.CustomDump.getOutfiles(logfile)
        if kwargs.get(jobutils.FLAG_CLEAN[1:]):
            jname = name.split(BaseTask.SEP)[0]
            for filename in outfiles.values():
                try:
                    os.remove(filename.replace(tname, jname))
                except FileNotFoundError:
                    pass
        outfiles = {x: [z.fn(y) for z in jobs] for x, y in outfiles.items()}
        jname = name.split(BaseTask.SEP)[0]
        inav = environutils.is_interactive()
        cls.DRIVER.CustomDump.combine(outfiles, log, jname, inav=inav)

    @classmethod
    def postAgg(cls, *jobs, name=None):
        """
        Report the status of the aggregation over all custom dump task output
        files.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param name: jobname based on which log file is found
        :type name: str
        :return: the label after job completion
        :rtype: str
        """

        jname = name.split(cls.SEP)[0]
        logfile = jname + logutils.DRIVER_LOG
        try:
            line = sh.grep(cls.RESULTS, logfile)
        except sh.ErrorReturnCode_1:
            return False
        line = line.strip().split('\n')
        lines = [x.split(cls.RESULTS)[-1].strip() for x in line]
        ext = '.' + cls.DRIVER.CustomDump.DATA_EXT.split('.')[-1]
        filenames = [x for x in lines if x.split()[-1].endswith(ext)]
        return f'{len(filenames)} files found'


class Lmp_Log(BaseTask):

    import lmp_log_driver as DRIVER
    READ_DATA = lammpsin.In.READ_DATA
    DATA_EXT = lammpsin.In.DATA_EXT
    RESULTS = DRIVER.LmpLog.RESULTS

    def setArgs(self):
        """
        Set the args for lmp log task.
        """
        super().setArgs()
        log_file = self.doc[self.KNOWN_ARGS][0]
        data_cmd = sh.grep(self.READ_DATA, log_file).split()
        data_file = [x for x in data_cmd if x.endswith(self.DATA_EXT)][0]
        args = [log_file, self.DRIVER.FLAG_DATA_FILE, data_file]
        args += list(self.doc[self.KNOWN_ARGS])[1:]
        self.doc[self.KNOWN_ARGS] = args

    @classmethod
    def aggregator(cls, *jobs, log=None, name=None, tname=None, **kwargs):
        """
        The aggregator job task that combines the lammps log analysis.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param log: the function to print user-facing information
        :type log: 'function'
        :param name: the jobname based on which output files are named
        :type name: str
        :param tname: aggregate the job tasks of this name
        :type tname: str
        """
        log(f"{len(jobs)} jobs found for aggregation.")
        job = jobs[0]
        logfile = job.fn(job.document[jobutils.OUTFILE][tname])
        outfiles = cls.DRIVER.LmpLog.getOutfiles(logfile)
        if kwargs.get(jobutils.FLAG_CLEAN[1:]):
            jname = name.split(BaseTask.SEP)[0]
            for filename in outfiles.values():
                try:
                    os.remove(filename.replace(tname, jname))
                except FileNotFoundError:
                    pass
        jobs = sorted(jobs, key=lambda x: x.statepoint[BaseTask.STATE_ID])
        outfiles = {x: [z.fn(y) for z in jobs] for x, y in outfiles.items()}
        jname = name.split(BaseTask.SEP)[0]
        inav = environutils.is_interactive()
        state_ids = [x.statepoint[BaseTask.STATE_ID] for x in jobs]
        state_label = kwargs.get('state_label')
        iname = pd.Index(state_ids, name=state_label) if state_label else None
        cls.DRIVER.LmpLog.combine(outfiles,
                                      log,
                                      jname,
                                      inav=inav,
                                      iname=iname)

    @classmethod
    def postAgg(cls, *jobs, name=None):
        """
        Report the status of the aggregation over all lmp log task output
        files.

        Main driver log should report results found the csv saved on the success
        of aggregation.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param name: jobname based on which log file is found
        :type name: str
        :return: the label after job completion
        :rtype: str
        """
        jname = name.split(cls.SEP)[0]
        logfile = jname + logutils.DRIVER_LOG
        try:
            line = sh.grep(cls.RESULTS, logfile)
        except sh.ErrorReturnCode_1:
            return False
        line = line.strip().split('\n')
        lines = [x.split(cls.RESULTS)[-1].strip() for x in line]
        ext = '.' + cls.DRIVER.LmpLog.DATA_EXT.split('.')[-1]
        filenames = [x for x in lines if x.split()[-1].endswith(ext)]
        return f'{len(filenames)} files found'
