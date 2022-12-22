import os
import pytest
import oplsua
import fragments
import testutils
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from contextlib import contextmanager

BUTANE_DATA = os.path.join('polym_builder', '_relax', 'data.polym')
BUTANE_DATA = testutils.test_file(BUTANE_DATA)


class TestOplsParser:

    @pytest.fixture
    def nprsr(self):
        return oplsua.OplsParser()

    @pytest.fixture
    def raw_prsr(self):
        raw_prsr = oplsua.OplsParser()
        raw_prsr.setRawContent()
        return raw_prsr

    def testSetRawContent(self, nprsr):
        nprsr.setRawContent()
        assert 10 == len(nprsr.raw_content)

    def testSetAtomType(self, raw_prsr):
        raw_prsr.setAtomType()
        assert 213 == len(raw_prsr.atoms)

    def testSetVdW(self, raw_prsr):
        raw_prsr.setVdW()
        assert 213 == len(raw_prsr.vdws)

    def testSetCharge(self, raw_prsr):
        raw_prsr.setCharge()
        assert 213 == len(raw_prsr.charges)

    def testSetBond(self, raw_prsr):
        raw_prsr.setBond()
        assert 150 == len(raw_prsr.bonds)

    def testSetAngle(self, raw_prsr):
        raw_prsr.setAngle()
        assert 306 == len(raw_prsr.angles)

    def testSetImproper(self, raw_prsr):
        raw_prsr.setImproper()
        assert 75 == len(raw_prsr.impropers)

    def testSetDihedral(self, raw_prsr):
        raw_prsr.setDihedral()
        assert 628 == len(raw_prsr.dihedrals)


class TestDataFileReader:

    @pytest.fixture
    def dfr(self):
        return oplsua.DataFileReader(BUTANE_DATA)

    def testRead(self, dfr):
        dfr.read()
        assert 1152 == len(dfr.lines)
        assert 11 == len(dfr.mk_idxes)

    def testSetDescription(self, dfr):
        dfr.read()
        dfr.setDescription()
