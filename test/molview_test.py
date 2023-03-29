import os
import sys
import pytest
from rdkit import Chem
from nemd import oplsua
from nemd import molview
from nemd import testutils

from unittest import mock


class TestTransConformer(object):
    @pytest.fixture
    def frm_vw(self):
        filepath = os.path.join('polym_builder', 'cooh123.data')
        datafile = testutils.test_file(filepath)
        data_reader = oplsua.DataFileReader(datafile)
        data_reader.run()
        frm_vw = molview.FrameView(data_reader)
        return frm_vw

    def testSetData(self, frm_vw):
        frm_vw.setData()
        assert (30, 6) == frm_vw.data.shape

    def testScatters(self, frm_vw):
        frm_vw.setData()
        frm_vw.scatters()
        assert 7 == len(frm_vw.fig['data'])

    def testLines(self, frm_vw):
        frm_vw.setData()
        frm_vw.scatters()
        frm_vw.lines()
        assert 61 == len(frm_vw.fig['data'])
