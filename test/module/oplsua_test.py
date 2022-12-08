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


class TestFragMol:

    @pytest.fixture
    def nprsr(self):
        return oplsua.OPLS_Parser()

    @pytest.fixture
    def raw_prsr(self):
        raw_prsr = oplsua.OPLS_Parser()
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