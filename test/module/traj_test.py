import os
import traj
import pytest
import fragments
import testutils
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from contextlib import contextmanager

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
        return traj.DistanceCell(frm)

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetSpan(self, dcell):
        dcell.setSpan()
        assert (dcell.span == 48).all()

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetgrids(self, dcell):
        dcell.setSpan()
        dcell.setgrids()
        assert (dcell.grids == 2).all()

    @pytest.mark.parametrize(('filename'), [(CC3COOH)])
    def testSetNeighborIds(self, dcell):
        dcell.setSpan()
        dcell.setgrids()
        dcell.setNeighborIds()
        assert 311 == len(dcell.neigh_ids)
