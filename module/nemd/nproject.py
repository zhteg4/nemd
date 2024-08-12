from flow.project import FlowProject

from nemd import jobutils


class FlowProject(FlowProject):

    OUTFILE = jobutils.OUTFILE
    OUTFILES = jobutils.OUTFILES
    MESSAGE = jobutils.MESSAGE

    def __init__(self, *args, jobname=None, **kwargs):
        """
        This is the constructor of the FlowProject class.
        :jobname str: the job name of the workflow.
        """
        super().__init__(*args, **kwargs)
        self.doc[jobutils.FLAG_JOBNAME] = jobname
        self.doc[self.MESSAGE] = self.doc.get(self.MESSAGE, {})

    @property
    def jobname(self):
        """
        Returns the job name of the current job.
        :return str: the job name of the current worflow.
        """
        return self.doc.get(jobutils.FLAG_JOBNAME, None)

    def open_job(self, *args, **kwargs):
        job = super().open_job(*args, **kwargs)
        job.doc[self.OUTFILE] = job.doc.get(self.OUTFILE, {})
        job.doc[self.OUTFILES] = job.doc.get(self.OUTFILES, {})
        job.doc[self.MESSAGE] = job.doc.get(self.MESSAGE, {})
        return job
