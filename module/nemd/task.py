import re
import os
import sh
import types
import argparse
import functools
import collections
import humanfriendly
import pandas as pd
from datetime import timedelta
from types import SimpleNamespace
from flow import FlowProject, aggregator

from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import lammpsin
from nemd import analyzer

FILE = jobutils.FILE


class BaseJob(logutils.Base):

    ARGS = jobutils.ARGS
    FLAG_JOBNAME = jobutils.FLAG_JOBNAME
    DATA_EXT = '.csv'
    PRE_RUN = None

    def __init__(self, job, name=None, driver=None, logger=None, **kwargs):
        """
        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param name: the jobname of this subjob, which is usually different from
            the workflow jobname
        :type name: str
        :param driver: imported driver module
        :type driver: 'module'
        :param logger:  print to this logger
        :type logger: 'logging.Logger'
        """
        super().__init__(logger=logger)
        self.job = job
        self.name = name
        self.driver = driver
        self.logger = logger
        self.doc = self.job.document
        self.args = list(map(str, self.doc.get(self.ARGS, [])))


class Job(BaseJob):
    """
    The class to setup a run_nemd driver job in a workflow.

    NOTE: this base one is a cmd job.
    """

    SPECIAL_CHAR_RE = re.compile("[@!#%^&*()<>?|}{:]")
    QUOTED_RE = re.compile('^".*"$|^\'.*\'$')
    PRE_RUN = jobutils.RUN_NEMD
    SEP = symbols.SPACE
    PREREQ = jobutils.PREREQ
    OUTFILE = jobutils.OUTFILE

    def run(self):
        """
        Main method to setup the job.
        """
        self.setArgs()
        self.removeUnkArgs()
        self.setName()
        self.addQuote()

    def setArgs(self):
        """
        Set arguments.
        """
        pre_jobs = self.doc[self.PREREQ].get(self.name)
        if pre_jobs is None:
            return
        try:
            args = self.driver.ARGS_TMPL[:]
        except AttributeError:
            return
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for pre_job in pre_jobs:
            index = args.index(FILE)
            args[index] = self.doc[self.OUTFILE][pre_job]
        self.args = args + self.args

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
        jobutils.set_arg(self.args, self.FLAG_JOBNAME, self.name)

    def addQuote(self):
        """
        Add quotes for str with special characters.
        """
        self.args = [self.quoteArg(x) for x in self.args]

    @classmethod
    def quoteArg(self, arg):
        """
        Quote the unquoted argument that contains special characters.
        """
        if self.SPECIAL_CHAR_RE.search(arg) and not self.QUOTED_RE.match(arg):
            return f"'{arg}'"
        return arg

    def getCmd(self, pre_cmd=None, extra_args=None, write=True):
        """
        Get command line str.

        :param pre_cmd list: the pre-command to run before the args
        :param extra_args list: extra args for the specific task
        :param write bool: the msg to be printed
        :return str: the command as str
        """

        if pre_cmd is None:
            pre_cmd = []
            if self.PRE_RUN:
                pre_cmd.append(self.PRE_RUN)
            if self.driver:
                pre_cmd.append(self.driver.PATH)

        if extra_args is None:
            extra_args = []

        cmd = self.SEP.join(map(str, pre_cmd + self.args + extra_args))

        if write:
            with open(f"{self.name}_cmd", 'w') as fh:
                fh.write(cmd)
        return cmd

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        outfiles = self.doc.get(self.OUTFILE, {})
        outfile = outfiles.get(self.name)
        return bool(outfile)


class AggJob(BaseJob):
    """
    The class to run an aggregator job in a workflow.

    NOTE: this base one is a non-cmd job.
    """

    TIME_BREAKDOWN = 'Task timing breakdown:'

    def __init__(self, *jobs, **kwargs):
        """
        :param jobs: the signac job instances to aggregate
        :type jobs: 'list' of 'signac.contrib.job.Job'
        """
        super().__init__(jobs[0], **kwargs)
        self.jobs = jobs
        self.project = self.job.project
        self.jobname = self.name.split(symbols.POUND_SEP)[0]

    def run(self):
        """
        Main method to run the aggregator job.
        """
        self.log(self.TIME_BREAKDOWN)
        info = collections.defaultdict(list)
        for job in self.jobs:
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
            self.log(f"{tname}: {', '.join(deltas)}; {ave} (ave)")
        self.project.doc[self.name] = False

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.name in self.project.doc

    @functools.cache
    def groupJobs(self):
        """
        Group jobs by the statepoints so that the jobs within one group only
        differ by the FLAG_SEED.

        return list of (pandas.Series, 'signac.job.Job') tuples: state point
            parameters, grouped jobs
        """
        jobs = collections.defaultdict(list)
        series = {}
        for job in self.jobs:
            statepoint = dict(job.statepoint)
            statepoint.pop(jobutils.FLAG_SEED, None)
            params = {
                x[1:] if x.startswith('-') else x: y
                for x, y in statepoint.items()
            }
            params = pd.Series(params).sort_index()
            key = params.to_csv(lineterminator=' ', sep=' ', header=False)
            series[key] = params
            jobs[key].append(job)
        keys = sorted(series.keys())
        for idx, key in enumerate(keys):
            series[key].index.name = idx
        return [tuple([series[x], jobs[x]]) for x in keys]


class BaseTask:
    """
    The task base class.
    """

    JobClass = Job
    AggClass = AggJob
    DRIVER = None

    ARGS = jobutils.ARGS
    OUTFILE = jobutils.OUTFILE
    FINISHED = jobutils.FINISHED
    DRIVER_LOG = logutils.DRIVER_LOG

    @classmethod
    def pre(cls, *args, **kwargs):
        """
        Set and check pre-conditions before starting the job.

        :return bool: True if the pre-conditions are met
        """
        return True

    @classmethod
    def operator(cls, *args, **kwargs):
        """
        The main opterator (function) for a job task executed after
        pre-conditions are met.

        :return str: the command to run a task.
        """
        obj = cls.JobClass(*args, driver=cls.DRIVER, **kwargs)
        obj.run()
        if hasattr(obj, 'getCmd'):
            return obj.getCmd()

    @classmethod
    def post(cls, *args, **kwargs):
        """
        The job is considered finished when the post-conditions return True.

        :return bool: True if the post-conditions are met
        """
        obj = cls.JobClass(*args, **kwargs)
        return obj.post()

    @classmethod
    def aggregator(cls, *args, **kwargs):
        """
        The aggregator job task.
        """
        obj = cls.AggClass(*args, driver=cls.DRIVER, **kwargs)
        obj.run()

    @classmethod
    def postAgg(cls, *args, **kwargs):
        """
        Post-condition for aggregator task.

        :return bool: True if the post-conditions are met
        """
        obj = cls.AggClass(*args, driver=cls.DRIVER, **kwargs)
        return obj.post()

    @classmethod
    def getOpr(cls,
               cmd=None,
               with_job=True,
               name=None,
               attr='operator',
               pre=False,
               post=None,
               aggregator=None,
               logger=None,
               **kwargs):
        """
        Duplicate and return the operator with jobname and decorators.

        NOTE: post-condition must be provided so that the current job can check
        submission eligibility using post-condition of its prerequisite job.
        On the other hand, the absence of pre-condition means the current is
        eligible for submission as long as its prerequisite jobs are completed.

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
        :param logger:  print to this logger
        :type logger: 'logging.Logger'
        :return: the operation to execute
        :rtype: 'function'
        :raise ValueError: the function cannot be found
        """
        if cmd is None:
            cmd = hasattr(cls.JobClass, 'getCmd')
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
        if logger:
            kwargs['logger'] = logger
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
        return func

    @classmethod
    def getAgg(cls,
               cmd=False,
               with_job=False,
               name=None,
               attr='aggregator',
               post=None,
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
        :param post: add post-condition for the aggregator if True
        :type post: bool
        :return: the operation to execute
        :rtype: 'function'
        """
        if post is None:
            post = cls.postAgg
        name = f"{name}{symbols.POUND_SEP}agg"
        return cls.getOpr(aggregator=aggregator(),
                          cmd=cmd,
                          with_job=with_job,
                          name=name,
                          attr=attr,
                          post=post,
                          **kwargs)

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


class Lammps(BaseTask):

    import lammps_driver as DRIVER


class LogJob(Job):

    READ_DATA = lammpsin.In.READ_DATA

    def setArgs(self):
        """
        Set arguments to analyze the log file.
        """
        super().setArgs()
        self.args = self.args[:1] + self.getDatafile() + self.args[1:]

    def getDatafile(self):
        """
        Get the data file from the log file.

        :return list: the list of arguments to add the data file
        """
        cmd = sh.grep(self.READ_DATA, self.args[0]).split()
        data_file = [x for x in cmd if x.endswith(lammpsin.In.DATA_EXT)][0]
        return data_file


class LogJobAgg(AggJob):

    FLAG_TASK = jobutils.FLAG_TASK

    def __init__(self, *args, **kwargs):
        """
        :param name: e.g. 'cb_lmp_log_#_lmp_log' is parsed as {jobname}_#_{name}
        """
        super().__init__(*args, **kwargs)
        self.tasks = jobutils.get_arg(self.args, self.FLAG_TASK, first=False)
        inav = jobutils.get_arg(self.args, jobutils.FLAG_INTERACTIVE)
        wdir = os.path.relpath(self.project.workspace, self.project.path)
        self.options = SimpleNamespace(jobname=self.project.jobname,
                                       interactive=inav,
                                       id=None,
                                       dir=wdir,
                                       name=self.jobname,
                                       jobs=None)

    def run(self):
        """
        Main method to run the aggregator job.
        """
        self.log(f"{len(self.jobs)} jobs found for aggregation.")
        for task in self.tasks:
            anlz = analyzer.Agg(task=task,
                                jobs=self.groupJobs(),
                                options=self.options,
                                logger=self.logger)
            anlz.run()
        self.project.doc[self.name] = False


class Lmp_Log(BaseTask):

    import lmp_log_driver as DRIVER
    JobClass = LogJob
    AggClass = LogJobAgg


class TrajJob(LogJob):

    DUMP = lammpsin.In.DUMP
    CUSTOM_EXT = lammpsin.In.CUSTOM_EXT

    def setArgs(self):
        """
        Set arguments to analyze the custom dump file.
        """
        super().setArgs()
        self.args[0] = self.trajfile

    @property
    def traj_file(self):
        """
        Return the trajectory file from the log file.

        :return str: the trajectory file.
        """
        cmd = sh.grep(self.DUMP, self.args[0]).split()
        traj_file = [x for x in cmd if x.endswith(self.CUSTOM_EXT)][0]
        return traj_file


class Lmp_Traj(Lmp_Log):

    import lmp_traj_driver as DRIVER
    JobClass = TrajJob
