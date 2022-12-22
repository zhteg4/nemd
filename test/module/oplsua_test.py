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

BUTANE_DATA = testutils.test_file(os.path.join('polym_builder',
                                               'cooh123.data'))


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
    def raw_dfr(self):
        return oplsua.DataFileReader(BUTANE_DATA)

    @pytest.fixture
    def dfr(self):
        dfr = oplsua.DataFileReader(BUTANE_DATA)
        dfr.read()
        dfr.setDescription()
        return dfr

    @pytest.fixture
    def df_reader(self):
        df_reader = oplsua.DataFileReader(BUTANE_DATA)
        df_reader.run()
        return df_reader

    def testRead(self, raw_dfr):
        raw_dfr.read()
        assert 210 == len(raw_dfr.lines)
        assert 11 == len(raw_dfr.mk_idxes)

    def testSetDescription(self, raw_dfr):
        raw_dfr.read()
        raw_dfr.setDescription()
        assert 5 == len(raw_dfr.struct_dsp)
        assert 5 == len(raw_dfr.dype_dsp)
        assert 3 == len(raw_dfr.box_dsp)

    def testSetMasses(self, dfr):
        dfr.setMasses()
        assert 8 == len(dfr.masses)

    def testSetAtoms(self, dfr):
        dfr.setMasses()
        dfr.setAtoms()
        assert 30 == len(dfr.atoms)

    def testSetMols(self, dfr):
        dfr.setMasses()
        dfr.setAtoms()
        dfr.setMols()
        assert 3 == len(dfr.mols)

    def testSetBonds(self, dfr):
        dfr.setBonds()
        assert 27 == len(dfr.bonds)

    def testAngles(self, dfr):
        dfr.setAngles()
        assert 31 == len(dfr.angles)

    def testSetDihedrals(self, dfr):
        dfr.setDihedrals()
        assert 30 == len(dfr.dihedrals)

    def testSetImpropers(self, dfr):
        dfr.setImpropers()
        assert 7 == len(dfr.impropers)

    @pytest.mark.parametrize(('include14', 'num'), [(True, 9), (False, 5)])
    def testSetClashExclusion(self, df_reader, include14, num):
        df_reader.setClashExclusion(include14=include14)
        assert num == len(df_reader.excluded[7])

    def testSetPairCoeffs(self, dfr):
        dfr.setPairCoeffs()
        assert 8 == len(dfr.vdws)

    def testSetVdwRadius(self, df_reader):
        df_reader.setPairCoeffs()
        df_reader.setVdwRadius()
        assert 30 == len(len(df_reader.radii))
