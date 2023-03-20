import os
import pytest
import numpy as np
from nemd import traj
from nemd import testutils

TRAJS = 'trajs'
BASE_DIR = testutils.test_file(TRAJS)
CC3COOH = os.path.join(BASE_DIR, 'CC3COOH.custom')
CC3COOH_RANDOMIZED = os.path.join(BASE_DIR, 'CC3COOH_randomized.custom')


class TestTraj:
    @pytest.fixture
    def raw_frms(self, filename):
        return traj.Frame.read(filename)

    @pytest.mark.parametrize(('filename', 'same'),
                             [(CC3COOH, False), (CC3COOH_RANDOMIZED, True)])
    def testRead(self, raw_frms, same):
        frms = list(raw_frms)
        assert 2 == len(frms)
        frm1, frm2 = frms
        assert same == all((frm1 == frm2).all())

    @pytest.fixture
    def frm(self, filename):
        return next(traj.Frame.read(filename))

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testInit(self, frm):
        array = frm.values
        nfrm = traj.Frame(array)
        assert all((frm == nfrm).all())


class TestDistanceCell:
    @pytest.fixture
    def dcell(self, filename):
        frm = next(traj.Frame.read(filename))
        return traj.DistanceCell(frm, cut=3., resolution=1.)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetSpan(self, dcell):
        dcell.setSpan()
        assert (dcell.span == 48).all()

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetgrids(self, dcell):
        dcell.setSpan()
        dcell.setgrids()
        assert (dcell.grids == 1).all()

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetNeighborIds(self, dcell):
        dcell.setSpan()
        dcell.setgrids()
        dcell.setNeighborIds()
        assert 311 == len(dcell.neigh_ids)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetAtomCell(self, dcell):
        dcell.setSpan()
        dcell.setgrids()
        dcell.setNeighborIds()
        dcell.setAtomCell()
        assert (48, 48, 48, 19) == dcell.atom_cell.shape

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testGetNeighbors(self, dcell):
        dcell.cut = 3
        dcell.resolution = 1
        dcell.setSpan()
        dcell.setgrids()
        dcell.setNeighborIds()
        dcell.setAtomCell()
        xyzs = [dcell.frm.getXYZ(x) for x in dcell.getNeighbors((0, 0, 0))]
        dists = [np.linalg.norm(x) for x in xyzs]
        assert 16 == len(dists)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testGetNeighbors(self, dcell):
        dcell.cut = 3
        dcell.resolution = 1
        dcell.setSpan()
        dcell.setgrids()
        dcell.setNeighborIds()
        dcell.setAtomCell()
        row = dcell.frm.getXYZ(1)
        assert not dcell.getClashes(row)
