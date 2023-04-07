import os
import pytest
from nemd import oplsua
from nemd import molview
from nemd import testutils


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

    def testSetScatters(self, frm_vw):
        frm_vw.setData()
        frm_vw.setEleSz()
        frm_vw.setScatters()
        assert 7 == len(frm_vw.markers)

    def testSetLines(self, frm_vw):
        frm_vw.setData()
        frm_vw.setEleSz()
        frm_vw.setScatters()
        frm_vw.setLines()
        assert 54 == len(frm_vw.lines)
