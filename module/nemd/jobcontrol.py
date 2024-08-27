import re
import sys
import flow
import itertools
import collections
import numpy as np
import pandas as pd
import networkx as nx

from nemd import task
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import fileutils


class Runner:
    """
    The main class to setup a workflow.
    """

    WORKSPACE = 'workspace'
    ARGS = jobutils.ARGS
    PREREQ = jobutils.PREREQ
    COMPLETED = 'completed'
    OPERATIONS = 'operations'
    JOB_ID = 'job_id'
    FLAG_SEED = jobutils.FLAG_SEED
    MESSAGE = jobutils.MESSAGE
    AGG_NAME_EXT = f"{symbols.POUND_SEP}agg"

    def __init__(self, options, argv, logger=None):
        """
        :param options: parsed commandline options
        :type options: 'argparse.Namespace'
        :param argv: list of commandline arguments
        :type argv: list
        :param logger: print to this logger if exists
        :type logger: 'logging.Logger'
        """
        self.options = options
        self.argv = argv
        self.logger = logger
        self.state = {}
        self.jobs = []
        self.oprs = {}
        self.classes = {}
        self.project = None
        self.agg_project = None
        self.prereq = collections.defaultdict(list)
        # flow/project.py gets logger from logging.getLogger(__name__)
        logutils.createModuleLogger('flow.project', file_ext=fileutils.LOG)

    def run(self):
        """
        The main method to run the integration tests.

        The linear pipline handles three things on request:
        1) clean previous projects
        2) run a project with task jobs
        3) run a project with aggregator jobs
        """
        if jobutils.TASK in self.options.jtype:
            self.setJob()
            self.setProject()
            self.setState()
            self.addJobs()
            self.cleanJobs()
            self.runJobs()
            self.logStatus()
            self.logMessage()
        if jobutils.AGGREGATOR in self.options.jtype:
            self.setAggJobs()
            self.setAggProject()
            self.cleanAggJobs()
            self.runAggJobs()

    def setJob(self):
        """
        Set the tasks for the job.
        """
        raise NotImplementedError('This method adds operators as job tasks. ')

    def setOpr(self, TaskCLass, name=None, agg=False, **kwargs):
        """
        Set one operations for the job.

        :param TaskCLass: the task class
        :type TaskCLass: 'task.Task' (sub)-class
        :param name: the name of the operation
        :type name: str
        :return: the name of the operation
        """
        if name is None:
            words = re.findall('[A-Z][^A-Z]*', TaskCLass.__name__)
            name = '_'.join([x.lower() for x in words])
        if agg:
            name = f"{name}{self.AGG_NAME_EXT}"
            self.oprs[name] = TaskCLass.getAgg(name=name, **kwargs)
            self.classes[name] = TaskCLass.AggClass
            return name
        self.oprs[name] = TaskCLass.getOpr(name=name, **kwargs)
        self.classes[name] = TaskCLass.JobClass
        return name

    def setAgg(self, *args, **kwargs):
        """
        Set one aggregatorfor the job.
        """
        return self.setOpr(*args,
                           **kwargs,
                           agg=True,
                           logger=self.logger,
                           options=self.options)

    def setProject(self):
        """
        Initiate the project.
        """
        self.project = flow.project.FlowProject.init_project()

    def setState(self):
        """
        Set the state flags and values.
        """
        try:
            seed_incre = np.arange(self.options.state_num)
        except AttributeError:
            return
        seed = int(jobutils.pop_arg(self.argv, self.FLAG_SEED, 0))
        self.state = {self.FLAG_SEED: list(map(str, seed + seed_incre))}

    def addJobs(self):
        """
        Add jobs to the project.

        NOTE:  _StatePointDict warns NumpyConversionWarning if statepoint dict
        contains numerical data types.
        """
        argvs = [[[x, z] for z in y] for x, y in self.state.items()]
        for argv in itertools.product(*argvs):
            # e.g. arg = (['-seed', '0'], ['-scale_factor', '0.95'])
            job = self.project.open_job(dict(tuple(x) for x in argv))
            job.document[self.ARGS] = self.argv[:] + sum(argv, [])
            job.document.update({self.PREREQ: self.prereq})
            self.jobs.append(job)

    def cleanJobs(self):
        """
        The post functions of the pre-job return False after the clean so that
        the job can run again on request.
        """
        if not self.options.clean:
            return
        for name, JobClass in self.classes.items():
            if name.endswith(self.AGG_NAME_EXT):
                continue
            for job in self.jobs:
                JobClass(job, name=name).clean()

    def runJobs(self):
        """
        Run all jobs registered in the project.
        """
        if not self.options.debug:
            self.project.run(np=self.options.cpu)
            return

        import matplotlib
        obackend = matplotlib.get_backend()
        backend = obackend if self.options.interactive else 'Agg'
        matplotlib.use(backend)
        import matplotlib.pyplot as plt
        depn = np.asarray(self.project.detect_operation_graph())
        graph = nx.DiGraph(depn)
        pos = nx.spring_layout(graph)
        names = [x for x in self.project.operations.keys()]
        labels = {key: name for (key, name) in zip(range(len(names)), names)}
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        nx.draw_networkx(graph, pos, ax=ax, labels=labels)
        if self.options.interactive:
            print("Showing task workflow graph. Click X to close the figure "
                  "and continue..")
            plt.show(block=True)
        fig.savefig(self.options.jobname + '_nx.png')
        self.project.run(np=self.options.cpu)

    def logStatus(self):
        """
        Look into each job and report the status.
        """
        status_file = self.options.jobname + fileutils.STATUS_LOG
        with open(status_file, 'w') as fh:
            # Fetching status and Fetching labels are printed to err handler
            self.project.print_status(detailed=True, file=fh, err=fh)
        # Log job status
        jobs = self.project.find_jobs()
        status = [self.project.get_job_status(x) for x in jobs]
        ops = [x[self.OPERATIONS] for x in status]
        completed = [all([y[self.COMPLETED] for y in x.values()]) for x in ops]
        failed_num = len([x for x in completed if not x])
        self.log(f"{len(jobs) - failed_num} / {len(jobs)} completed jobs.")
        if not failed_num:
            return
        id_ops = []
        for completed, op, stat, in zip(completed, ops, status):
            if completed:
                continue
            failed_ops = [x for x, y in op.items() if not y[self.COMPLETED]]
            id_ops.append([stat[self.JOB_ID], ', '.join(reversed(failed_ops))])
        id_ops = pd.DataFrame(id_ops, columns=[self.JOB_ID, 'operations'])
        id_ops.set_index(self.JOB_ID, inplace=True)
        self.log(id_ops.to_markdown())

    def logMessage(self):
        """
        Log the messages from the jobs.
        """
        jobs = self.project.find_jobs()
        ops = [self.project.get_job_status(x)[self.OPERATIONS] for x in jobs]
        completed = [all(y[self.COMPLETED] for y in x.values()) for x in ops]
        if not any(completed):
            return
        jobs = [x for x, y in zip(jobs, completed) if y]
        fjobs = [x for x in jobs if any(x.doc.get(self.MESSAGE, {}).values())]
        self.log(f"{len(jobs) - len(fjobs)} / {len(jobs)} succeeded jobs.")
        if not fjobs:
            return
        func = lambda x: '\n'.join(f"{k}: {v}" for k, v in x.items() if v)
        data = {self.MESSAGE: [func(x.doc[self.MESSAGE]) for x in fjobs]}
        fcn = lambda x: '\n'.join(f"{k.strip('-')}: {v}" for k, v in x.items())
        data['parameters'] = [fcn(x.statepoint) for x in fjobs]
        ids = pd.Index([x.id for x in fjobs], name=self.JOB_ID)
        info = pd.DataFrame(data, index=ids)
        self.log(info.to_markdown())

    def setAggJobs(self):
        """
        Collect jobs and analyze for statics, chemical space, and states.
        """
        pnames = [x for x in self.oprs.keys() if x.endswith(self.AGG_NAME_EXT)]
        name = self.setAgg(task.BaseTask, name=self.options.jobname)
        for pre_name in pnames:
            self.setPreAfter(pre_name, name)

    def setAggProject(self):
        """
        Initiate the aggregation project.
        """
        prj_path = self.project.path if self.project else self.options.prj_path
        try:
            self.agg_project = flow.project.FlowProject.get_project(prj_path)
        except LookupError as err:
            self.log_error(str(err))

    def cleanAggJobs(self):
        """
        Run aggregation project.
        """
        if not self.options.clean:
            return
        for name, JobClass in self.classes.items():
            if not name.endswith(self.AGG_NAME_EXT):
                continue
            JobClass(*self.jobs, name=name).clean()

    def runAggJobs(self):
        """
        Run aggregation project.
        """
        self.agg_project.run()

    def setPreAfter(self, pre, cur):
        """
        Set the prerequisite of a job.

        :param pre str: the operation name runs first
        :param cur str: the operation name who runs after the prerequisite job
        """
        if pre is None or cur is None:
            return
        flow.project.FlowProject.pre.after(self.oprs[pre])(self.oprs[cur])
        self.prereq[cur].append(pre)

    def log(self, msg, timestamp=False):
        """
        Print message to the logger or screen.

        :param msg: the message to print
        :type msg: str
        :param timestamp:
        :type timestamp: bool
        """
        if self.logger:
            logutils.log(self.logger, msg, timestamp=timestamp)
        else:
            print(msg)

    def log_error(self, msg):
        """
        Print this message and exit the program.

        :param msg str: the msg to be printed
        """
        self.log(msg + '\nAborting...', timestamp=True)
        sys.exit(1)
