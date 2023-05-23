import os
import sh
import types
import functools
import ordered_set
from flow import FlowProject

from nemd import logutils
from nemd import jobutils
from nemd import parserutils

FILE = "$FILE"


class BaseTask:
    """
    The task base class.
    """

    ID = 'id'
    TARGS = 'targs'
    STATE_ID = 'state_id'
    ARGS = jobutils.ARGS
    KNOWN_ARGS = jobutils.KNOWN_ARGS
    UNKNOWN_ARGS = jobutils.UNKNOWN_ARGS
    RUN_NEMD = jobutils.RUN_NEMD
    FINISHED = jobutils.FINISHED
    OUTFILE = jobutils.OUTFILE
    DRIVER_LOG = logutils.DRIVER_LOG
    PREREQ = jobutils.PREREQ

    def __init__(self, job, pre_run=RUN_NEMD, jobname=None):
        """
        :param job: the signac job instance
        :type job: 'signac.contrib.job.Job'
        :param pre_run: append this str before the driver path
        :type pre_run: str
        :param jobname: the jobname
        :type jobname: str
        """
        self.job = job
        self.pre_run = pre_run
        self.jobname = jobname
        self.doc = self.job.document
        self.run_driver = [self.DRIVER.PATH]
        if self.pre_run:
            self.run_driver = [self.pre_run] + self.run_driver

    def run(self):
        """
        The main method to run.
        """
        self.setArgs()
        self.setCmd()

    def setArgs(self):
        """
        Set known and unknown arguments.
        """
        parser = self.DRIVER.get_parser()
        args = list(self.doc.get(self.TARGS, {}).get(self.jobname, []))
        args += list(self.doc[self.ARGS])
        _, unknown = parser.parse_known_args(args)
        self.doc[self.UNKNOWN_ARGS] = unknown
        args = ordered_set.OrderedSet(args)
        self.doc[self.KNOWN_ARGS] = list(args.difference(unknown))

    def setName(self):
        """
        Set the jobname of the known args.
        """
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], jobutils.FLAG_JOBNAME,
                         self.jobname)

    def setCmd(self):
        """
        Set the command which the job will execute.
        """
        # self.doc[KNOWN_ARGS] is not a list but BufferedJSONAttrLists
        args = list(self.doc[self.KNOWN_ARGS])
        self.cmd = list(map(str, self.run_driver + args))

    def getCmd(self):
        """
        Get command line str.

        :return: the command as str
        :rtype: str
        """
        return ' '.join(self.cmd)

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
    def pre(cls, job, name):
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
            return True
        if not all(job.doc[cls.OUTFILE][x] for x in pre_jobs):
            # The job has incomplete prerequisite jobs and has to wait
            return False
        args = cls.DRIVER.ARGS_TMPL[:]
        # Pass the outfiles of the prerequisite jobs to the current via cmd args
        # Please rearrange or modify the prerequisite jobs' input by subclassing
        for pre in job.doc[cls.PREREQ][name]:
            index = args.index(FILE)
            args[index] = job.doc[cls.OUTFILE][pre]
        job.doc.setdefault(cls.TARGS, {})[name] = args
        return True

    @classmethod
    def post(cls, job, name):
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
        return job.document.get(cls.OUTFILE, {}).get(name)

    @staticmethod
    def operator(job):
        """
        The main opterator (function) for a job task executed after
        pre-conditions are met.
        """
        pass

    @classmethod
    def getOperator(cls, with_job=True, name=None):
        """
        Duplicate and return the operator with jobname and decorators.

        :param with_job: perform the execution in job dir with context management
        :type with_job: bool
        :param name: the jobname
        :type name: str
        :return: the operation to execute
        :rtype: 'function'
        """
        # Duplicate the function
        # Reference as http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
        origin_name = cls.operator.__name__
        if name is None:
            name = cls.operator.__name__
        cls.operator.__name__ = name
        func = types.FunctionType(cls.operator.__code__,
                                  cls.operator.__globals__,
                                  name=name,
                                  argdefs=cls.operator.__defaults__,
                                  closure=cls.operator.__closure__)
        func = functools.update_wrapper(func, cls.operator)
        func.__kwdefaults__ = cls.operator.__kwdefaults__
        cls.operator.__name__ = origin_name
        # Pass jobname and add FlowProject decorators
        func = functools.update_wrapper(functools.partial(func, jobname=name),
                                        func)
        func = FlowProject.operation(cmd=True, with_job=with_job,
                                     name=name)(func)
        func = FlowProject.post(lambda x: cls.post(x, name))(func)
        func = FlowProject.pre(lambda x: cls.pre(x, name))(func)
        return func


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
        """
        polymer_builder = Polymer_Builder(*arg, **kwargs)
        polymer_builder.run()
        return polymer_builder.getCmd()


class Lammps_Driver:
    """
    LAMMPS wrapper to package the imported software as a nemd driver.
    """

    LMP_SERIAL = 'lmp_serial'
    PATH = LMP_SERIAL
    JOBNAME = 'lammps'
    IN = '-in'

    ARGS_TMPL = [IN, FILE]

    @staticmethod
    def get_parser():
        """

        :return:
        :rtype: 'argparse.ArgumentParser'
        """
        parser = parserutils.get_parser(
            description='This is a customized parser wrapper for lammps.')
        parser.add_argument('-in',
                            metavar='IN_SCRIPT',
                            type=parserutils.type_file,
                            required=True,
                            help='Read input from this file.')
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
        Get the polymer builder operation command.
        """
        lmp = Lammps(*arg, **kwargs)
        lmp.run()
        return lmp.getCmd()
