import os
import pytest
from unittest import mock

from nemd import testutils
from nemd import itestutils

BASE_DIR = os.path.join(testutils.TEST_FILE_DIR, 'itest')


def get_job(tid=1, document=None, tdir=os.curdir):
    if document is None:
        document = {}
    statepoint = {itestutils.FLAG_DIR: os.path.join(BASE_DIR, f"{tid:0>4}")}
    fn = lambda x: os.path.join(tdir, x) if x else tdir
    return mock.Mock(statepoint=statepoint, document=document, fn=fn)


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
        return itestutils.CmdJob(get_job(), delay=True)

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
        tdir = os.path.join(BASE_DIR, 'ea8c25e09124635e93178c1725ae8ee7')
        return itestutils.Exist('amorp_bldr.data', job=get_job(tdir=tdir))

    def testRun(self, exist):
        exist.run()
        exist.job = get_job()
        with pytest.raises(FileNotFoundError):
            exist.run()
