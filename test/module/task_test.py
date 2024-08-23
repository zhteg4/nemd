import os
import sys
import shutil
import pytest
import datetime
from unittest import mock
from flow.project import FlowProject

from nemd import task
from nemd import jobutils
from nemd import environutils

TEST_DIR = environutils.get_test_dir()
if TEST_DIR is None:
    sys.exit("Error: test directory cannot be found.")
BASE_DIR = os.path.join(TEST_DIR, 'test_files', 'itest')
JOB_DIR = os.path.join(BASE_DIR, '6e4cfb3bcc2a689d42d099e51b9efe23')
PROJ_DIR = os.path.join(BASE_DIR, 'e053136e2cd7374854430c868b3139e1')


def get_jobs(proj_dir=PROJ_DIR):
    basename = os.path.basename(proj_dir)
    shutil.copytree(proj_dir, basename)
    proj = FlowProject.get_project(basename)
    return proj.find_jobs()


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
    import lammps_driver as DRIVER

    @pytest.fixture
    def job(self, tmp_dir):
        shutil.copyfile(os.path.join(JOB_DIR, self.INFILE), self.INFILE)
        job = jobutils.Job(job_dir=JOB_DIR)
        return task.Job(job, name='lammps_runner', driver=self.DRIVER)

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
        job.outfile = 'output.log'
        assert job.post() is True


class TestAggJob:

    @pytest.fixture
    def agg(self, tmp_dir):
        jobs = get_jobs()
        agg = task.AggJob(*jobs, logger=mock.Mock())
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


class TestMol_Bldr:

    @pytest.fixture
    def job(self):
        job = jobutils.Job()
        job.doc[jobutils.ARGS] = ['[Ar]']
        return job

    @pytest.fixture
    def task(self):
        return task.Mol_Bldr()

    def testPre(self, task):
        assert task.pre() is True

    def testOperator(self, task, job, tmp_dir):
        assert task.operator(
            job) == 'run_nemd mol_bldr_driver.py [Ar] -JOBNAME job'

    def testPost(self, task, job):
        assert task.post(job) is False
        job.doc[jobutils.OUTFILE] = {'job': 'wa.log'}
        assert task.post(job) is True

    def testGetOpr(self, task, job):
        opr = task.getOpr(name='mol')
        assert opr._flow_cmd is True


class TestLmp_Traj:

    @pytest.fixture
    def task(self):
        return task.Lmp_Log()

    def testAggPost(self, task):
        job = jobutils.Job()
        assert task.aggPost(job) is False
        job.project.doc['message'] = {'logjobagg': False}
        assert task.aggPost(job) is True

    def testGetAgg(self, task, tmp_dir):
        agg = task.getAgg(get_jobs())
        assert agg._flow_cmd is True
