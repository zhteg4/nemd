import os
import pytest
from nemd import traj
from nemd import oplsua
from nemd import molview
from nemd import testutils

COOH123 = testutils.test_file(os.path.join('polym_builder', 'cooh123.data'))
NACL_DATA = testutils.test_file(os.path.join('trajs', 'NaCl.data'))
NACL_CUSTOM = testutils.test_file(os.path.join('trajs', 'NaCl.custom'))
CC_DATA = testutils.test_file(os.path.join('trajs', 'CC.data'))
CC_CUSTOM = testutils.test_file(os.path.join('trajs', 'CC.custom'))


class TestTransConformer(object):

    @pytest.fixture
    def frm_vw(self, datafile):
        data_reader = oplsua.DataFileReader(datafile)
        data_reader.run()
        frm_vw = molview.FrameView(data_reader)
        return frm_vw

    @pytest.fixture
    def frm_df(self, datafile, traj_path):
        data_reader = oplsua.DataFileReader(datafile)
        data_reader.run()
        frm_df = molview.FrameView(data_reader)
        frm_df.setData()
        frm_df.setEleSz()
        frm_df.setScatters()
        frm_df.setLines()
        frm_df.addTraces()
        frms = traj.get_frames(traj_path)
        frm_df.setFrames(frms)
        frm_df.updateLayout()
        return frm_df

    @pytest.fixture
    def frm_trj(self, traj_path):
        frm_df = molview.FrameView()
        frms = traj.get_frames(traj_path)
        frm_df.setFrames(frms)
        frm_df.updateLayout()
        return frm_df

    @pytest.mark.parametrize(('datafile'), [(COOH123)])
    def testSetData(self, frm_vw):
        frm_vw.setData()
        assert (30, 6) == frm_vw.data.shape

    @pytest.mark.parametrize(('datafile'), [(COOH123)])
    def testSetScatters(self, frm_vw):
        frm_vw.setData()
        frm_vw.setEleSz()
        frm_vw.setScatters()
        assert 7 == len(frm_vw.markers)

    @pytest.mark.parametrize(('datafile'), [(COOH123)])
    def testSetLines(self, frm_vw):
        frm_vw.setData()
        frm_vw.setEleSz()
        frm_vw.setScatters()
        frm_vw.setLines()
        assert 54 == len(frm_vw.lines)

    @pytest.mark.parametrize(('datafile', 'traj_path', 'data_num', 'frm_num'),
                             [(NACL_DATA, NACL_CUSTOM, 2, 4),
                              (CC_DATA, CC_CUSTOM, 3, 4)])
    def testAll_DataAndTraj(self, frm_df, data_num, frm_num):
        # frm_df.show(); pdb.set_trace(); to view in the browser
        assert data_num == len(frm_df.fig.data)
        assert frm_num == len(frm_df.fig.frames)

    @pytest.mark.parametrize(('traj_path', 'data_num', 'frm_num'),
                             [(NACL_CUSTOM, 13, 4), (CC_CUSTOM, 13, 4)])
    def testAll_Traj(self, frm_trj, data_num, frm_num):
        # frm_df.show(); pdb.set_trace(); to view in the browser
        assert data_num == len(frm_trj.fig.data)
        assert frm_num == len(frm_trj.fig.frames)
