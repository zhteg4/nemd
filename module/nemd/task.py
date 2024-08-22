import re
import os
import sh
import types
import functools
import collections
import pandas as pd
from types import SimpleNamespace
from flow import FlowProject, aggregator

from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import lammpsin
from nemd import analyzer
from nemd import timeutils

FILE = jobutils.FILE


class BaseJob(logutils.Base):
    """
    The base class for all jobs in the workflow. This class can be subclassed
    to create cmd and non-cmd jobs depending on whether the job returns a cmd
    to run in the shell or not. In terms of the workflow, the subclassed jobs
    can be used as normal task jobs or aggregate jobs.
    """

    ARGS = jobutils.ARGS
    FLAG_JOBNAME = jobutils.FLAG_JOBNAME
    MESSAGE = jobutils.MESSAGE
    DATA_EXT = '.csv'
    PRE_RUN = None

    def __init__(self, job, name='base', driver=None, logger=None, **kwargs):
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

    @property
    def message(self):
        """
        The message of the job.

        :return str: the message of the job.
        """
        return self.doc.get(self.MESSAGE, {}).get(self.name)

    @message.setter
    def message(self, value):
        """
        Set message of the job.

        :value str: the message of the job.
        """
        if self.MESSAGE not in self.doc:
            self.doc[self.MESSAGE] = {}
        self.doc[self.MESSAGE].update({self.name: value})


class Job(BaseJob):
    """
    The class to set up a run_nemd driver job in a workflow.

    NOTE: this is a cmd job.
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

    def getCmd(self, pre_cmd=None, extra_args=None, write=True, name=None):
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

        if name is None:
            name = self.name

        cmd = self.SEP.join(map(str, pre_cmd + self.args + extra_args))

        if write:
            with open(f"{name}_cmd", 'w') as fh:
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

    NOTE: this is a non-cmd job.
    """

    MS_FMT = '%M:%S'
    MS_LMT = '59:59'
    DELTA_LMT = timeutils.str2delta(MS_LMT, fmt=MS_FMT)
    MANE = symbols.NAME
    TIME = symbols.TIME.lower()
    ID = symbols.ID

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
        info = []
        for job in self.jobs:
            for filename in job.doc.get(jobutils.LOGFILE, {}).values():
                log = logutils.LogReader(job.fn(filename))
                log.run()
                info.append([log.options.default_name, log.task_time, job.id])
        info = pd.DataFrame(info, columns=[self.MANE, self.TIME, self.ID])
        # Group the jobs by the labels
        data, grouped = {}, info.groupby(self.MANE)
        for key, dat in sorted(grouped, key=lambda x: x[1].size, reverse=True):
            val = dat.drop(columns=self.MANE)
            val.sort_values(self.TIME, ascending=False, inplace=True)
            ave = val.time.mean()
            ave = pd.DataFrame([[ave, 'ave']], columns=[self.TIME, self.ID])
            val = pd.concat([ave, val]).reset_index(drop=True)
            val = val.apply(lambda x: f'{self.delta2str(x.time)} {x.id[:3]}',
                            axis=1)
            data[key[:8]] = val
        data = pd.DataFrame(data)
        total_time = timeutils.delta2str(info.time.sum())
        self.log(logutils.LogReader.TOTOAL_TIME + total_time)
        self.log(data.fillna('').to_markdown(index=False))
        self.message = False

    @property
    def message(self):
        """
        The message of the agg job.

        :return str: the message of the job.
        """
        return self.project.doc.get(self.MESSAGE, {}).get(self.name)

    @message.setter
    def message(self, value):
        """
        Set message of the agg job.

        :value str: the message of the job.
        """
        if self.MESSAGE not in self.doc:
            self.project.doc[self.MESSAGE] = {}
        self.project.doc[self.MESSAGE].update({self.name: value})

    @classmethod
    def delta2str(cls, delta):
        """
        Delta time to string with upper limit.

        :param delta: the time delta object
        :type delta: 'datetime.timedelta'
        :return str: the string representation of the time delta with upper limit
        """
        if delta > cls.DELTA_LMT:
            return cls.MS_LMT
        return timeutils.delta2str(delta, fmt=cls.MS_FMT)

    def post(self):
        """
        The job is considered finished when the post-conditions return True.

        :return: True if the post-conditions are met.
        """
        return self.name in self.project.doc[self.MESSAGE]

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
    The task base class holding task job, aggregator job, and driver module.

    The post method is used to check if the job is finished, and a True return
    tasks the job out of the queue without executing it.
    The pre method is used to check if the job is eligible for submission. The
    current job is submitted only when its pre method returns the True and the
    post methods of all its prerequisite jobs return True as well.
    The operator method is used to execute the job, and is called on execution.
    The aggregator method is used to aggregate the results of the task jobs.
    """

    JobClass = Job
    AggClass = AggJob
    DRIVER = None

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
        if not hasattr(obj, 'getCmd'):
            return
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


class Mol_Bldr(BaseTask):

    import mol_bldr_driver as DRIVER


class Amorp_Bldr(BaseTask):

    import amorp_bldr_driver as DRIVER


class Xtal_Bldr(BaseTask):

    import xtal_bldr_driver as DRIVER


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
        return [self.driver.FLAG_DATA_FILE, data_file]


class LogJobAgg(AggJob):

    FLAG_TASK = jobutils.FLAG_TASK

    def __init__(self, *args, **kwargs):
        """
        :param name: e.g. 'cb_lmp_log_#_lmp_log' is parsed as {jobname}_#_{name}
        """
        super().__init__(*args, **kwargs)
        self.tasks = jobutils.get_arg(self.args, self.FLAG_TASK, first=False)
        if self.tasks is None:
            self.tasks = self.driver.DEFAULT_TASKS
        self.tasks = [x.lower() for x in self.tasks]
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
        self.message = False


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
        self.args[0] = self.traj_file

    @property
    def traj_file(self):
        """
        Return the trajectory file from the log file.

        :return str: the trajectory file.
        """
        cmd = sh.grep(self.DUMP, self.args[0]).split()
        return [x for x in cmd if x.endswith(self.CUSTOM_EXT)][0]


class Lmp_Traj(Lmp_Log):

    import lmp_traj_driver as DRIVER
    JobClass = TrajJob
