from flow.project import FlowProject

from nemd import jobutils


class FlowProject(FlowProject):

    OUTFILE = jobutils.OUTFILE
    OUTFILES = jobutils.OUTFILES

    def __init__(self, *args, jobname=None, **kwargs):
        """
        This is the constructor of the FlowProject class.
        :jobname str: the job name of the workflow.
        """
        super().__init__(*args, **kwargs)
        self.doc[jobutils.FLAG_JOBNAME] = jobname

    @property
    def jobname(self):
        """
        Returns the job name of the current job.
        :return str: the job name of the current worflow.
        """
        return self.doc.get(jobutils.FLAG_JOBNAME, None)

    def open_job(self, *args, **kwargs):
        job = super().open_job(*args, **kwargs)
        job.doc[jobutils.OUTFILE] = job.doc.get(jobutils.OUTFILE, {})
        job.doc[jobutils.OUTFILES] = job.doc.get(jobutils.OUTFILES, {})
        return job
