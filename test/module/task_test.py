import os
import sys
import shutil
import pytest
import datetime
from unittest import mock

from nemd import task
from nemd import jobutils
from nemd import environutils
from nemd.nproject import FlowProject

TEST_DIR = environutils.get_test_dir()
if TEST_DIR is None:
    sys.exit("Error: test directory cannot be found.")
BASE_DIR = os.path.join(TEST_DIR, 'test_files', 'itest')
JOB_DIR = os.path.join(BASE_DIR, '6e4cfb3bcc2a689d42d099e51b9efe23')


class TestBaseJob:

    @pytest.fixture
    def job(self):
        return task.BaseJob(jobutils.Job())

    def testMessage(self, job):
        assert job.message is None
        job.message = False
        assert job.message is False


class TestJob:

    INFILE = 'amorphous_builder.in'

    @pytest.fixture
    def job(self, tmp_dir):
        shutil.copyfile(os.path.join(JOB_DIR, self.INFILE), self.INFILE)
        job = jobutils.Job(job_dir=JOB_DIR)
        import lammps_driver as DRIVER
        return task.Job(job, name='lammps_runner', driver=DRIVER)

    def testSetArgs(self, job):
        job.setArgs()
        assert job.args == ['amorphous_builder.in', '[Ar]', '-seed', '0']

    def testRemoveUnkArgs(self, job):
        job.setArgs()
        job.removeUnkArgs()
        assert job.args == ['amorphous_builder.in']

    def testSetName(self, job):
        job.setName()
        assert job.args[-2:] == ['-JOBNAME', 'lammps_runner']

    def testAddQuote(self, job):
        job.args = ['*']
        job.addQuote()
        assert job.args[0] == "'*'"

    def testGetCmd(self, job):
        job.args = ['amorphous_builder.in']
        cmd = job.getCmd()
        assert cmd == 'run_nemd lammps_driver.py amorphous_builder.in'

    def testPost(self, job):
        assert job.post() is False
        job.doc[jobutils.OUTFILE][job.name] = 'output.log'
        assert job.post() is True


class TestAggJob:

    PROJ_DIR = os.path.join(BASE_DIR, 'e053136e2cd7374854430c868b3139e1')

    @pytest.fixture
    def agg(self, tmp_dir):
        basename = os.path.basename(self.PROJ_DIR)
        shutil.copytree(self.PROJ_DIR, basename)
        proj = FlowProject.get_project(basename, jobname='ab_lmp_traj')
        import lmp_traj_driver as DRIVER
        jobs = proj.find_jobs()
        agg = task.AggJob(*jobs, driver=DRIVER, logger=mock.Mock())
        return agg

    def testPost(self, agg):
        assert agg.post() is False
        agg.message = False
        assert agg.post() is True

    def testGroupJobs(self, agg):
        jobs = agg.groupJobs()
        assert len(jobs) == 1
        assert len(jobs[0]) == 2

    def testRun(self, agg):
        agg.run()
        assert agg.logger.info.called

    @pytest.mark.parametrize("delta, expected",
                             [(datetime.timedelta(hours=3), '59:59'),
                              (datetime.timedelta(minutes=3), '03:00')])
    def testDelta2str(self, delta, expected):
        assert expected == task.AggJob.delta2str(delta)
