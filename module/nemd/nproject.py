from flow.project import FlowProject

from nemd import jobutils


class FlowProject(FlowProject):

    OUTFILE = jobutils.OUTFILE
    OUTFILES = jobutils.OUTFILES

    def open_job(self, *args, **kwargs):
        job = super().open_job(*args, **kwargs)
        job.doc[jobutils.OUTFILE] = job.doc.get(jobutils.OUTFILE, {})
        job.doc[jobutils.OUTFILES] = job.doc.get(jobutils.OUTFILES, {})
        return job
