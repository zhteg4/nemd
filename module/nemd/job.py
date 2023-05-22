import os
import sh
import ordered_set
from flow import FlowProject

from nemd import logutils
from nemd import jobutils


class BaseJob:

    ID = 'id'
    NAME = 'name'
    TASK_ID = 'task_id'
    STATE_ID = 'state_id'
    ARGS = jobutils.ARGS
    KNOWN_ARGS = jobutils.KNOWN_ARGS
    UNKNOWN_ARGS = jobutils.UNKNOWN_ARGS
    RUN_NEMD = jobutils.RUN_NEMD
    FINISHED = jobutils.FINISHED
    DRIVER_LOG = logutils.DRIVER_LOG

    def __init__(self, job, pre_run=RUN_NEMD, jobname=None):
        self.job = job
        self.jobname = jobname
        self.doc = self.job.document
        self.run_driver = [self.DRIVER.PATH]
        if pre_run:
            self.run_driver = [pre_run] + self.run_driver
        self.task_id = None

    def run(self):
        self.setTaskId()
        self.setArgs()
        self.setCmd()

    def setTaskId(self):
        self.task_id = self.doc.get(self.TASK_ID, 0)
        self.doc.update({self.TASK_ID: str(self.task_id + 1)})

    def setArgs(self):
        parser = self.DRIVER.get_parser()
        _, unknown = parser.parse_known_args(self.doc[self.ARGS])
        self.doc[self.UNKNOWN_ARGS] = unknown
        args = ordered_set.OrderedSet(self.doc[self.ARGS])
        self.doc[self.KNOWN_ARGS] = list(args.difference(unknown))

    def setName(self):
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], jobutils.FLAG_JOBNAME,
                         self.jobname)

    def setCmd(self):
        # self.doc[KNOWN_ARGS] is not a list but BufferedJSONAttrLists
        args = list(self.doc[self.KNOWN_ARGS])
        self.cmd = list(map(str, self.run_driver + args))

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

    # @classmethod
    # def pre(cls, job):
    #     # import pdb; pdb.set_trace()
    #     # task_id = job.statepoint().get(cls.TASK_ID, 0)
    #     # job.update_statepoint({cls.TASK_ID: task_id + 1})
    #     return True

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

    @classmethod
    def post(cls, job, name):
        return job.document.get(jobutils.OUTFILE, {}).get(name)

    @staticmethod
    def operator(job):
        pass

    @classmethod
    def getOperator(cls, with_job=True, name=None):
        func = cls.operator
        if name:
            func.__name__ = name
        func = FlowProject.operation(cmd=True, with_job=with_job)(func)
        func = FlowProject.post(lambda x: cls.post(x, name))(func)
        return func


class Polymer_Builder(BaseJob):

    import polymer_builder_driver as DRIVER
    FLAG_SEED = jobutils.FLAG_SEED

    # def __init__(self, job):
    #     super().__init__(job)

    def run(self):
        super().run()
        self.setSeed()

    def setSeed(self):
        seed = jobutils.get_arg(self.doc[self.KNOWN_ARGS], self.FLAG_SEED, 0)
        state = self.job.statepoint()
        seed = int(seed) + int(state.get(self.STATE_ID, state.get(self.ID)))
        jobutils.set_arg(self.doc[self.KNOWN_ARGS], self.FLAG_SEED, seed)

    @staticmethod
    def operator(job):
        """
        Build cell.

        :param job 'signac.contrib.job.Job': the job object
        :return str: the shell command to execute
        """
        polymer_builder = Polymer_Builder(job, jobname=__name__)
        polymer_builder.run()
        return polymer_builder.getCmd()


class Lammps_Driver:
    LMP_SERIAL = 'lmp_serial'
    PATH = LMP_SERIAL
    JOBNAME = 'lammps'


class Lammps_Runner(BaseJob):

    DRIVER = Lammps_Driver

    def __init__(self, *args, pre_run=None, **kwargs):
        super().__init__(*args, pre_run=pre_run, **kwargs)

    def run(self):
        super().run()

    def setArgs(self, input='*.in'):
        import pdb
        pdb.set_trace()
        parser = self.DRIVER.get_parser()
        _, unknown = parser.parse_known_args(self.doc[self.ARGS])
        self.doc[self.UNKNOWN_ARGS] = unknown
        args = ordered_set.OrderedSet(self.doc[self.ARGS])
        self.doc[self.KNOWN_ARGS] = list(args.difference(unknown))

    @staticmethod
    def operator(job):
        """
        Build cell.

        :param job 'signac.contrib.job.Job': the job object
        :return str: the shell command to execute
        """
        lammps_runner = Lammps_Runner(job)
        lammps_runner.run()
        return lammps_runner.getCmd()

    @classmethod
    def post(cls, job):
        import pdb
        pdb.set_trace()
