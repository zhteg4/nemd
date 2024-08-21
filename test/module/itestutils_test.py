import os
import sys
import pytest

from nemd import itestutils
from nemd import environutils

TEST_DIR = environutils.get_test_dir()
if TEST_DIR is None:
    sys.exit("Error: test directory cannot be found.")
BASE_DIR = os.path.join(TEST_DIR, 'test_files', 'itest')


class TestCmd:

    @pytest.fixture
    def cmd(self):
        return itestutils.Cmd(os.path.join(BASE_DIR, '0001'), delay=True)

    def testRead(self, cmd):
        cmd.read()
        assert len(cmd.args) == 2

    def testSetComment(self, cmd):
        cmd.read()
        cmd.setComment()
        assert cmd.comment == 'Amorphous builder on C'


class Job:

    def __init__(self, tid=1, document=None, tdir=os.curdir):
        self.statepoint = {
            itestutils.FLAG_DIR: os.path.join(BASE_DIR, f"{tid:0>4}")
        }
        self.document = document if document else {}
        self.dir = tdir

    def fn(self, x):
        return os.path.join(self.dir, x) if self.dir else self.dir


JOB = Job(tdir=os.path.join(BASE_DIR, 'ea8c25e09124635e93178c1725ae8ee7'))


class TestCmdJob:

    @pytest.fixture
    def job(self):
        return itestutils.CmdJob(JOB, delay=True)

    def testParse(self, job):
        job.parse()
        assert job.comment == 'Amorphous builder on C'

    def testSetName(self, job):
        job.parse()
        job.setName()
        assert job.args[0].endswith('amorp_bldr')

    def testAddQuote(self, job):
        job.args = ['run_nemd amorp_bldr_driver.py C(C)']
        job.addQuote()
        assert job.args[0] == "run_nemd amorp_bldr_driver.py 'C(C)'"

    def testGetCmd(self, job):
        job.run()
        cmd = job.getCmd()
        assert cmd

    def testPost(self, job):
        assert not job.post()
        job.doc['outfile'] = {'amorp_bldr': 'amorp_bldr.data'}
        assert job.post()


class TestExist:

    @pytest.fixture
    def exist(self):
        return itestutils.Exist('amorp_bldr.data', job=JOB)

    def testRun(self, exist):
        try:
            exist.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"
        exist.targets[0] = os.path.join(BASE_DIR, 'amorp_bldr.data')
        with pytest.raises(FileNotFoundError):
            exist.run()


class TestNot_Exist:

    @pytest.fixture
    def not_exist(self):
        return itestutils.Not_Exist('amorp_bldr.data', job=JOB)

    def testRun(self, not_exist):
        with pytest.raises(FileNotFoundError):
            not_exist.run()
        not_exist.targets[0] = os.path.join(BASE_DIR, 'amorp_bldr.data')
        try:
            not_exist.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"


class TestCmp:

    @pytest.fixture
    def cmp(self):
        original = os.path.join(BASE_DIR, '0001', 'polymer_builder.data')
        return itestutils.Cmp(original, 'amorp_bldr.data', job=JOB)

    def testRun(self, cmp):
        try:
            cmp.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"
        cmp.job = Job()
        try:
            cmp.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"
