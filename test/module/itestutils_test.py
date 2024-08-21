import os
import types
import pytest

from nemd import testutils
from nemd import itestutils
from nemd import environutils

BASE_DIR = os.path.join(testutils.TEST_FILE_DIR, 'itest')


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
        document={}
        statepoint = {itestutils.FLAG_DIR: os.path.join(BASE_DIR, '0001')}
        job = types.SimpleNamespace(statepoint=statepoint, document=document)
        return itestutils.CmdJob(job, delay=True)

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
