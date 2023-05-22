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

    def __init__(self, options, argv, jobname, logger=None):
        """
        :param options 'argparse.Namespace': parsed commandline options.
        :param jobname str: the jobname
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

    def log(self, msg, timestamp=False):
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
            job.init()

    def runProject(self):
        """
        Run all jobs registered in the project
        """
        # import pdb;pdb.set_trace()
        # set_trace
        # import numpy as np
        # import networkx as nx
        # from matplotlib import pyplot as plt
        #
        # project = self.project
        # ops = project.operations.keys()
        # adj = np.asarray(project.detect_operation_graph())
        #
        # plt.figure()
        # g = nx.DiGraph(adj)
        # pos = nx.spring_layout(g)
        # nx.draw_networkx(g, pos)
        # import pdb;pdb.set_trace()
        # nx.draw_networkx_labels(
        #     g, pos,
        #     labels={key: name for (key, name) in
        #             zip(range(len(ops)), [o for o in ops])})
        #
        # plt.show()
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
        # [all([y['completed'] for y in x['operations'].values()]) for x in
        #  status]
