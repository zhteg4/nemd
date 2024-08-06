import sys
import shutil
import collections
import numpy as np
import networkx as nx
from nemd.nproject import FlowProject

from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import fileutils
from nemd.task import BaseTask


class Runner:
    """
    The main class to run integration tests.
    """

    STATE_ID = 'state_id'
    WORKSPACE = 'workspace'
    FLOW_PROJECT = 'flow.project'
    ARGS = jobutils.ARGS
    PREREQ = jobutils.PREREQ
    COMPLETED = 'completed'
    OPERATIONS = 'operations'
    JOB_ID = 'job_id'
    MSG = symbols.MSG

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
        self.project = None
        self.status_file = self.options.jobname + fileutils.STATUS_LOG
        # flow/project.py gets logger from logging.getLogger(__name__)
        logutils.createModuleLogger(self.FLOW_PROJECT, file_ext=fileutils.LOG)
        self.agg_project = None
        self.status_fh = None
        self.prereq = collections.defaultdict(list)

    def run(self):
        """
        The main method to run the integration tests.

        The linear pipline handles three things on request:
        1) clean previous projects
        2) run a project with task jobs
        3) run a project with aggregator jobs
        """
        with open(self.status_file, 'w') as self.status_fh:
            if jobutils.TASK in self.options.jtype:
                self.cleanJobs()
                self.setJob()
                self.setProject()
                self.addJobs()
                self.runJobs()
                self.logStatus()
            if jobutils.AGGREGATOR in self.options.jtype:
                self.setAggJobs()
                self.setAggProject()
                self.cleanAggJobs()
                self.runAggJobs()

    def cleanJobs(self):
        """
        Remove the previous task results on request.
        """
        if not self.options.clean:
            return
        if jobutils.TASK in self.options.jtype:
            try:
                shutil.rmtree(self.WORKSPACE)
            except FileNotFoundError:
                pass
            return

    def setJob(self):
        """
        Set the tasks for the job.

        Must be over-written by subclass and called before init_project()
        so that functions can be registered via decoration.
        """
        raise NotImplementedError('This method adds operators as job tasks. ')

    def setProject(self):
        """
        Initiate the project.
        """
        self.project = FlowProject.init_project()

    def addJobs(self, ids=None):
        """
        Add jobs to the project.

        :param ids list: the job ids based on which state points are set.
        """
        ids = range(self.options.state_num) if ids is None else ids
        for idx in ids:
            # _StatePointDict warns NumpyConversionWarning if state_id not str
            job = self.project.open_job({self.STATE_ID: str(idx)})
            job.doc[jobutils.OUTFILE] = job.doc.get(jobutils.OUTFILE, {})
            job.doc[jobutils.OUTFILES] = job.doc.get(jobutils.OUTFILES, {})
            job.document[self.ARGS] = self.argv[:]
            job.document.update({self.PREREQ: self.prereq})

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
        # Fetching status and Fetching labels are printed to err handler
        self.project.print_status(detailed=True,
                                  file=self.status_fh,
                                  err=self.status_fh)
        jobs = self.project.find_jobs()
        status = [self.project.get_job_status(x) for x in jobs]
        ops = [x[self.OPERATIONS] for x in status]
        completed = [all([y[self.COMPLETED] for y in x.values()]) for x in ops]
        failed_num = len([x for x in completed if not x])
        self.log(f"{len(jobs) - failed_num} / {len(jobs)} completed jobs.")
        if not failed_num:
            return
        for job, completed in zip(jobs, completed):
            if completed:
                continue
            failed_tasks = [x for x, y in ops.items() if not y[self.COMPLETED]]
            failed_tasks = ', '.join(reversed(failed_tasks))
            labels = ', '.join([x for x in self.project.labels(job)])
            self.log(f"Failed tasks are {failed_tasks}. (label: {labels}, "
                     f"id: {status[self.JOB_ID]}")

    def setAggJobs(self):
        """
        Collect jobs and analyze for statics, chemical space, and states.
        """
        BaseTask.getAgg(logger=self.logger, name=self.options.jobname)

    def setAggProject(self):
        """
        Initiate the aggregation project.
        """
        prj_path = self.project.path if self.project else self.options.prj_path
        try:
            self.agg_project = FlowProject.get_project(prj_path)
        except LookupError as err:
            self.log_error(str(err))

    def cleanAggJobs(self):
        """
        Run aggregation project.
        """
        if not self.options.clean:
            return
        if not jobutils.AGGREGATOR in self.options.jtype:
            return
        for name in self.agg_project.operations.keys():
            self.agg_project.doc.pop(name)

    def runAggJobs(self):
        """
        Run aggregation project.
        """
        self.agg_project.run()

    def setPrereq(self, cur, pre):
        """
        Set the prerequisite of a job.

        :param cur: the operation (function) who runs after the prerequisite job
        :type cur: 'function'
        :param pre: the operation (function) runs first
        :type pre: 'function'
        """
        FlowProject.pre.after(pre)(cur)
        self.prereq[cur.__name__].append(pre.__name__)

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