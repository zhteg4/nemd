import os
import sys
import types
import pytest
import contextlib

from nemd import jobutils
from nemd import itestutils
from nemd import environutils

TEST_DIR = environutils.get_test_dir()
if TEST_DIR is None:
    sys.exit("Error: test directory cannot be found.")
BASE_DIR = os.path.join(TEST_DIR, 'test_files', 'itest')
JOB_DIR = os.path.join(BASE_DIR, 'ea8c25e09124635e93178c1725ae8ee7')


class Job(jobutils.Job):

    def __init__(self, tid=1, idir=BASE_DIR, job_dir=JOB_DIR):
        super().__init__(job_dir)
        self.tid = tid
        self.idir = idir
        flag_dir = os.path.join(self.idir, f"{self.tid:0>4}")
        self.statepoint[itestutils.FLAG_DIR] = flag_dir


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


class TestCmdJob:

    @pytest.fixture
    def job(self):
        return itestutils.CmdJob(Job(job_dir=os.curdir), delay=True)

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

    def testGetCmd(self, job, tmp_dir):
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
        return itestutils.Exist('amorp_bldr.data', job=Job())

    def testRun(self, exist):
        try:
            exist.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"
        exist.targets[0] = os.path.join(BASE_DIR, 'amorp_bldr.data')
        with pytest.raises(FileNotFoundError):
            exist.run()


class TestNotExist:

    @pytest.fixture
    def not_exist(self):
        return itestutils.NotExist('amorp_bldr.data', job=Job())

    def testRun(self, not_exist):
        with pytest.raises(FileNotFoundError):
            not_exist.run()
        not_exist.targets[0] = os.path.join(BASE_DIR, 'amorp_bldr.data')
        try:
            not_exist.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"

class TestIn:

    @pytest.fixture
    def in_obj(self):
        return itestutils.In('Finished.', 'amorp_bldr-driver.log', job=Job())

    def testRun(self, in_obj):
        try:
            in_obj.run()
        except ValueError:
            assert False, "ValueError should not be raised"
        in_obj.strs = ['Aborted..']
        with pytest.raises(ValueError):
            in_obj.run()


class TestCmp:

    @pytest.fixture
    def cmp(self):
        return itestutils.Cmp('polymer_builder.data',
                              'amorp_bldr.data',
                              job=Job())

    def testRun(self, cmp):
        try:
            cmp.run()
        except FileNotFoundError:
            assert False, "FileNotFoundError should not be raised"
        cmp.targets[0] = cmp.targets[0].replace('polymer_builder.data', 'cmd')
        with pytest.raises(ValueError):
            cmp.run()


class TestCheck:

    @pytest.fixture
    def check(self):
        return itestutils.Check(job=Job(), delay=True)

    def testSetOperators(self, check):
        check.parse()
        check.setOperators()
        assert len(check.operators) == 1

    def testExecute(self, check):
        check.parse()
        check.setOperators()
        try:
            check.execute(check.operators[0])
        except KeyError:
            assert False, "KeyError should not be raised"
        with pytest.raises(KeyError):
            check.execute(['wa', 'polymer_builder.data', 'amorp_bldr.data'])


class TestCheckJob:

    @pytest.fixture
    def job(self):
        return itestutils.CheckJob(Job())

    def testRun(self, job):
        with contextlib.redirect_stdout(None):
            job.run()
        assert job.message is False

    def testPost(self, job):
        assert job.post() is False
        with contextlib.redirect_stdout(None):
            job.run()
        assert job.post() is True


class TestTag:

    @pytest.fixture
    def tag(self):
        return itestutils.Tag(job=Job())

    def testSetLogs(self, tag):
        tag.setLogs()
        assert len(tag.logs) == 1

    def testSetSlow(self, tag):
        tag.operators = []
        tag.setLogs()
        tag.setSlow()
        assert tag.operators[0][0] == 'slow'

    def testSetLabel(self, tag):
        tag.operators = []
        tag.setLogs()
        tag.setLabel()
        assert tag.operators[0][0] == 'label'

    def testWrite(self, tag, tmp_dir):
        tag.pathname = os.path.basename(tag.pathname)
        with contextlib.redirect_stdout(None):
            tag.write()
        assert os.path.exists('tag')

    def testSlow(self, tag):
        tag.options = types.SimpleNamespace(slow=None)
        assert tag.slow() is False
        tag.options = types.SimpleNamespace(slow=2.)
        assert tag.slow() is True

    def testLabeled(self, tag):
        tag.options = types.SimpleNamespace(label=None)
        assert tag.labeled() is True
        tag.options = types.SimpleNamespace(label=['wa'])
        assert tag.labeled() is False
        tag.options = types.SimpleNamespace(label=['wa', 'amorp_bldr'])
        assert tag.labeled() is True


class TestTagJob:

    @pytest.fixture
    def tag(self):
        return itestutils.TagJob(job=Job(idir=os.curdir))

    def testRun(self, tag, tmp_dir):
        os.mkdir('0001')
        with contextlib.redirect_stdout(None):
            tag.run()
        assert tag.message is False
