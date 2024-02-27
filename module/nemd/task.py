import os
import sh
import types
import logging
import functools
import collections
import humanfriendly
import pandas as pd
from datetime import timedelta
from flow import FlowProject, aggregator

from nemd import oplsua
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import parserutils
from nemd import environutils

FILE = jobutils.FILE

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):

    if logger is None:
        return
    logger.debug(msg)


class BaseTask:
    """
    The task base class.
    """

    ID = 'id'
    TIME = 'time'
    STIME = 'stime'
    ETIME = 'etime'
    STATE_ID = 'state_id'
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
        self.doc = self.job.document
        self.run_driver = [self.DRIVER.PATH]
        if self.pre_run:
            self.run_driver = [self.pre_run] + self.run_driver

    def run(self):
        """
        The main method to run.
        """
        self.setArgs()
        self.setName()

    def setArgs(self):
        """
        Set known and unknown arguments.
        """
        parser = self.DRIVER.get_parser()
        args = list(self.doc.get(self.TARGS, {}).get(self.name, []))
        args += list(self.doc[self.ARGS])
        _, unknown = parser.parse_known_args(args)
        self.doc[self.UNKNOWN_ARGS] = unknown
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

    def getCmd(self, write=True):
        """
        Get command line str.

        :return: the command as str
        :rtype: str
        """
        # self.doc[KNOWN_ARGS] is not a list but BufferedJSONAttrLists
        args = list(self.doc[self.KNOWN_ARGS])
        cmd = ' '.join(list(map(str, self.run_driver + args)))
        if write:
            with open(f"{self.name}_cmd", 'w') as fh:
                fh.write(cmd)
        return cmd

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
        return os.path.exists(logfile) and sh.tail('-2', logfile).startswith(
            cls.FINISHED)

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

    @staticmethod
    def operator(job):
        """
        The main opterator (function) for a job task executed after
        pre-conditions are met.
        """
        pass

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
        :param attr: the class method that is a operator function
        :type attr: str
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
        """
        if post is None:
            post = cls.post
        if pre is None:
            pre = cls.pre

        func = cls.DupeFunc(attr, name)
        # Pass jobname, taskname, and logging function
        kwargs.update({'name': name})
        if tname:
            kwargs['tname'] = tname
        if log:
            kwargs['log'] = log
        func = functools.update_wrapper(functools.partial(func, **kwargs),
                                        func)
        func = FlowProject.operation(cmd=cmd,
                                     with_job=with_job,
                                     name=name,
                                     aggregator=aggregator)(func)
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
    def DupeFunc(cls, attr, name):
        """
        Duplicate a function or static method with new naming.
        From http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)

        :param attr: the class method that is a operator function
        :type attr: str
        :param name: the taskname
        :type name: str
        :return: the function to execute
        :rtype: 'function'
        """
        ofunc = getattr(cls, attr)
        origin_name = ofunc.__name__
        if name is None:
            name = ofunc.__name__
        ofunc.__name__ = name
        func = types.FunctionType(ofunc.__code__,
                                  ofunc.__globals__,
                                  name=name,
                                  argdefs=ofunc.__defaults__,
                                  closure=ofunc.__closure__)
        func = functools.update_wrapper(func, ofunc)
        func.__kwdefaults__ = cls.operator.__kwdefaults__
        ofunc.__name__ = origin_name
        return func

    @staticmethod
    def aggregator(*jobs, log=None, **kwargs):
        """
        The aggregator job task to report the time cost of each task.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param log: the function to print user-facing information
        :type log: 'function'
        """

        delta_times = collections.defaultdict(list)
        for job in jobs:
            for tname, filename in job.doc[jobutils.LOGFILE].items():
                delta = logutils.get_time(job.fn(filename))
                delta_times[tname].append(delta)
        log(BaseTask.TIME_BREAKDOWN)
        for tname, deltas in delta_times.items():
            ave = sum(deltas, timedelta(0)) / len(deltas)
            deltas = [humanfriendly.format_timespan(x) for x in deltas]
            ave = humanfriendly.format_timespan(ave)
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
        Get and register an aggregator job task that collects custom dump task
        outputs.

        :param cmd: Whether the aggregator function returns a command to run
        :type cmd: bool
        :param with_job: Whether chdir to the job dir
        :type with_job: bool
        :param name: the name of this aggregator job task.
        :type name: str
        :param attr: the class method that is a aggregator function
        :type attr: str
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


class Polymer_Builder(BaseTask):

    import polymer_builder_driver as DRIVER
    FLAG_SEED = jobutils.FLAG_SEED

    def run(self):
        """
        The main method to run.
        """
        super().run()
        self.setSeed()

    def setSeed(self):
        """
        Set the random seed based on state id so that each task starts from a
        different state in phase space and the task collection can better
        approach the ergodicity.
        """
        seed = jobutils.get_arg(self.doc[self.KNOWN_ARGS], self.FLAG_SEED, 0)
        state = self.job.statepoint()
        seed = int(seed) + int(state.get(self.STATE_ID, state.get(self.ID)))
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], self.FLAG_SEED, seed)

    @staticmethod
    def operator(*arg, **kwargs):
        """
        Get the polymer builder operation command.

        :return str: the command to run a task.
        """
        polymer_builder = Polymer_Builder(*arg, **kwargs)
        polymer_builder.run()
        return polymer_builder.getCmd()


class Crystal_Builder(BaseTask):

    import crystal_builder_driver as DRIVER
    FLAG_SCALED_FACTOR = DRIVER.FLAG_SCALED_FACTOR

    def run(self):
        """
        The main method to run.
        """
        super().run()
        self.setScaledFactor()

    def setScaledFactor(self):
        """
        Set the scaled factor so that each cell starts from different vectors.
        """

        scaled_factor = jobutils.get_arg(self.doc[self.KNOWN_ARGS],
                                         self.FLAG_SCALED_FACTOR, 1)
        state = self.job.statepoint()
        scaled_factor = float(scaled_factor) * float(
            state.get(self.STATE_ID, state.get(self.ID)))
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], self.FLAG_SCALED_FACTOR,
                         scaled_factor)

    @staticmethod
    def operator(*arg, **kwargs):
        """
        Get the crystal builder operation command.

        :return str: the command to run a task.
        """
        xtal_builder = Crystal_Builder(*arg, **kwargs)
        xtal_builder.run()
        return xtal_builder.getCmd()


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

    ARGS_TMPL = [FLAG_IN, FILE, '-screen', 'none']

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

    @staticmethod
    def operator(*arg, **kwargs):
        """
        Get the lammps operation command.
        """
        lmp = Lammps(*arg, **kwargs)
        lmp.run()
        return lmp.getCmd()

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
    CUSTOM_EXT = oplsua.LammpsIn.CUSTOM_EXT
    DUMP = oplsua.LammpsIn.DUMP
    READ_DATA = oplsua.LammpsIn.READ_DATA
    DATA_EXT = oplsua.LammpsIn.DATA_EXT
    RESULTS = DRIVER.CustomDump.RESULTS

    @staticmethod
    def operator(*arg, **kwargs):
        """
        Get the polymer builder operation command.

        :return str: the command to run a task.
        """
        custom_dump = Custom_Dump(*arg, **kwargs)
        custom_dump.run()
        return custom_dump.getCmd()

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
        args = [dump_file, self.DRIVER.FlAG_DATA_FILE, data_file]
        args += list(self.doc[self.KNOWN_ARGS])[1:]
        self.doc[self.KNOWN_ARGS] = args

    @staticmethod
    def aggregator(*jobs, log=None, name=None, tname=None, **kwargs):
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
        outfiles = Custom_Dump.DRIVER.CustomDump.getOutfiles(logfile)
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
        Custom_Dump.DRIVER.CustomDump.combine(outfiles, log, jname, inav=inav)

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
    READ_DATA = oplsua.LammpsIn.READ_DATA
    DATA_EXT = oplsua.LammpsIn.DATA_EXT
    RESULTS = DRIVER.LmpLog.RESULTS

    @staticmethod
    def operator(*arg, **kwargs):
        """
        Get the polymer builder operation command.

        :return str: the command to run a task.
        """
        lmp_Log = Lmp_Log(*arg, **kwargs)
        lmp_Log.run()
        return lmp_Log.getCmd()

    def setArgs(self):
        """
        Set the args for custom dump task.
        """
        super().setArgs()
        log_file = self.doc[self.KNOWN_ARGS][0]
        data_cmd = sh.grep(self.READ_DATA, log_file).split()
        data_file = [x for x in data_cmd if x.endswith(self.DATA_EXT)][0]
        args = [log_file, self.DRIVER.FlAG_DATA_FILE, data_file]
        args += list(self.doc[self.KNOWN_ARGS])[1:]
        self.doc[self.KNOWN_ARGS] = args

    @staticmethod
    def aggregator(*jobs, log=None, name=None, tname=None, **kwargs):
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
        outfiles = Lmp_Log.DRIVER.LmpLog.getOutfiles(logfile)
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
        Lmp_Log.DRIVER.LmpLog.combine(outfiles,
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
