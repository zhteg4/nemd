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


def get_jobs(basename='e053136e2cd7374854430c868b3139e1'):
    proj_dir = os.path.join(BASE_DIR, basename)
    shutil.copytree(proj_dir, os.curdir, dirs_exist_ok=True)
    proj = FlowProject.get_project(os.curdir)
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

    @pytest.fixture
    def job(self, tmp_dir):
        infile = 'amorphous_builder.in'
        shutil.copyfile(os.path.join(JOB_DIR, infile), infile)
        job = jobutils.Job(job_dir=JOB_DIR)
        return task.Job(job, name='lammps_runner', driver=task.Lammps.DRIVER)

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


class TestLogJob:

    @pytest.fixture
    def job(self, tmp_dir):
        job_dir = os.path.join(BASE_DIR, '1c57f0964168565049315565b1388af9')
        fname = 'lammps_runner.log'
        shutil.copyfile(os.path.join(job_dir, fname), fname)
        job = jobutils.Job(job_dir=job_dir)
        return task.LogJob(job, name='lmp_log', driver=task.LmpLog.DRIVER)

    def testSetArgs(self, job):
        job.setArgs()
        assert job.args[1:3] == ['-data_file', 'crystal_builder.data']

    def testGetDatafile(self, job):
        job.args[0] = 'lammps_runner.log'
        data_file = job.getDataFile()
        assert data_file == ['-data_file', 'crystal_builder.data']


class TestTrajJob:

    @pytest.fixture
    def job(self, tmp_dir):
        job_dir = os.path.join(BASE_DIR, 'e053136e2cd7374854430c868b3139e1',
                               'workspace', '6e4cfb3bcc2a689d42d099e51b9efe23')
        fname = 'lammps_runner.log'
        shutil.copyfile(os.path.join(job_dir, fname), fname)
        job = jobutils.Job(job_dir=job_dir)
        return task.TrajJob(job, name='lmp_traj', driver=task.LmpTraj.DRIVER)

    def testSetArgs(self, job):
        job.setArgs()
        assert job.args[1:3] == ['-data_file', 'amorphous_builder.data']

    def testGetTrajfile(self, job):
        job.args[0] = 'lammps_runner.log'
        traj_file = job.getTrajFile()
        assert traj_file == 'dump.custom.gz'


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


class TestLogJobAgg:

    @pytest.fixture
    def agg(self, tmp_dir):
        jobs = get_jobs(basename='c1f776be48922ec50a6607f75c34c78f')
        return task.LogJobAgg(*jobs,
                              logger=mock.Mock(),
                              name='lmp_log_#_agg',
                              driver=task.LmpLog.DRIVER)

    def testRun(self, agg):
        assert agg.post() is False
        agg.run()
        assert agg.post() is True


class TestMolBldr:

    @pytest.fixture
    def job(self):
        job = jobutils.Job()
        job.doc[jobutils.ARGS] = ['[Ar]']
        return job

    @pytest.fixture
    def task(self):
        return task.MolBldr()

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


class TestLmpLog:

    @pytest.fixture
    def task(self):
        return task.LmpLog()

    def testAggPost(self, task):
        job = jobutils.Job()
        assert task.aggPost(job) is False
        job.project.doc['message'] = {'logjobagg': False}
        assert task.aggPost(job) is True

    def testGetAgg(self, task, tmp_dir):
        agg = task.getAgg(get_jobs())
        assert agg._flow_cmd is True
