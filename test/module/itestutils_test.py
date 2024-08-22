import os
import sys
import json
import pytest
import contextlib
import collections

from nemd import itestutils
from nemd import environutils

TEST_DIR = environutils.get_test_dir()
if TEST_DIR is None:
    sys.exit("Error: test directory cannot be found.")
BASE_DIR = os.path.join(TEST_DIR, 'test_files', 'itest')


class Job:

    def __init__(self, tid=1, doc=None, tdir=os.curdir):
        self.tid = tid
        self.doc = doc
        self.dir = tdir
        flag_dir = os.path.join(BASE_DIR, f"{self.tid:0>4}")
        self.statepoint = {itestutils.FLAG_DIR: flag_dir}
        if self.doc is None:
            self.doc = collections.defaultdict(dict)
        self.document = self.doc
        if self.dir is None:
            return
        job_doc = os.path.join(self.dir, 'signac_job_document.json')
        if not os.path.isfile(job_doc):
            return
        with open(job_doc, 'r') as fh:
            self.doc = json.load(fh).copy()
        self.document = self.doc

    def fn(self, x):
        return os.path.join(self.dir, x) if self.dir else self.dir


def get_job(basename='ea8c25e09124635e93178c1725ae8ee7'):
    if basename is None:
        return Job()
    return Job(tdir=os.path.join(BASE_DIR, basename))


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
        return itestutils.CmdJob(get_job(basename=None), delay=True)

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
        return itestutils.Exist('amorp_bldr.data', job=get_job())

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
        return itestutils.Not_Exist('amorp_bldr.data', job=get_job())

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
        return itestutils.Cmp('polymer_builder.data',
                              'amorp_bldr.data',
                              job=get_job())

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
        return itestutils.Check(job=get_job(), delay=True)

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
        return itestutils.CheckJob(get_job())

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
        return itestutils.Tag(job=get_job())

    def testSetLogs(self, tag):
        tag.setLogs()
        assert len(tag.logs) == 1
