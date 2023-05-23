import collections
import shutil
from flow import FlowProject

from nemd import logutils
from nemd import jobutils
from nemd import fileutils


class Runner:
    """
    The main class to run integration tests.
    """

    STATE_ID = 'state_id'
    WORKSPACE = 'workspace'
    FLOW_PROJECT = 'flow.project'
    ARGS = jobutils.ARGS
    PREREQ = jobutils.PREREQ

    def __init__(self, options, argv, jobname, logger=None):
        """
        :param options: parsed commandline options
        :type options: 'argparse.Namespace'
        :param argv: list of commandline arguments
        :type argv: list
        :param jobname: the jobname
        :type jobname: str
        :param logger: print to this logger if exists
        :type logger: 'logging.Logger'
        """
        self.options = options
        self.argv = argv
        self.jobname = jobname
        self.logger = logger
        self.project = None
        self.status_file = self.jobname + fileutils.STATUS_LOG
        # flow/project.py gets logger from logging.getLogger(__name__)
        logutils.createModuleLogger(self.FLOW_PROJECT, file_ext=fileutils.LOG)
        self.status_fh = None
        self.prereq = collections.defaultdict(list)

    def run(self):
        """
        The main method to run the integration tests.
        """
        with open(self.status_file, 'w') as self.status_fh:
            self.clean()
            # Decorators register functions to project (before init_project)
            self.setTasks()
            self.setProject()
            self.addJobs()
            self.runProject()
            self.logStatus()

    def log(self, msg, timestamp=False):
        """
        Print message to the logger or screen.

        :param msg: the message to print
        :type msg: str
        :param timestamp:
        :type timestamp: bool
        :return:
        :rtype:
        """
        if self.logger:
            logutils.log(self.logger, msg, timestamp=timestamp)
        else:
            print(msg)

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

    def setTasks(self):
        """
        Set the tasks for the job.

        Must be over-written by subclass.
        """
        raise NotImplementedError('This method adds operators as job tasks. ')

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
            job = self.project.open_job({self.STATE_ID: id})
            job.document[self.ARGS] = self.argv[:]
            job.document.update({self.PREREQ: self.prereq})
            job.init()

    def runProject(self):
        """
        Run all jobs registered in the project.
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
        completed = [
            all([y['completed'] for y in x['operations'].values()])
            for x in status
        ]
        self.log(f"{len(completed)} / {len(status)} completed.")
