import os
import sys
import shutil
import pytest
from nemd import task
from nemd import jobutils
from nemd import environutils

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

    @pytest.fixture
    def job(self):
        job = jobutils.Job(job_dir=JOB_DIR)
        import lammps_driver as DRIVER
        return task.Job(job, name='lammps_runner', driver=DRIVER)

    def testSetArgs(self, job):
        job.setArgs()
        assert job.args == ['amorphous_builder.in', '[Ar]', '-seed', '0']

    def testRemoveUnkArgs(self, job, tmp_dir):
        infile = 'amorphous_builder.in'
        shutil.copyfile(os.path.join(JOB_DIR, infile), infile)
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

    def testGetCmd(self, job, tmp_dir):
        job.args = ['amorphous_builder.in']
        cmd = job.getCmd()
        assert cmd == 'run_nemd lammps_driver.py amorphous_builder.in'

    def testPost(self, job):
        assert job.post() is False
        job.doc[jobutils.OUTFILE][job.name] = 'output.log'
        assert job.post() is True
