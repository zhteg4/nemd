import re
import os
import pytest
import numpy as np
from rdkit import Chem

from nemd import oplsua
from nemd import lammpsdata
from nemd import parserutils


class TestConformer:

    @pytest.fixture
    def conf(self):
        mol = lammpsdata.Mol.MolFromSmiles('CCCC(C)C')
        mol.EmbedMolecule()
        return mol.GetConformer()

    def testAtoms(self, conf):
        assert conf.atoms.shape == (6, 7)

    def testBonds(self, conf):
        assert conf.bonds.shape == (5, 3)

    def testAngles(self, conf):
        assert conf.angles.shape == (4, 4)

    def testDihedrals(self, conf):
        assert conf.dihedrals.shape == (3, 5)

    def testImpropers(self, conf):
        assert conf.impropers.shape == (1, 5)


class TestMol:

    @pytest.fixture
    def mol(self):
        return lammpsdata.Mol.MolFromSmiles('[H]OC(=O)CC', delay=True)

    def testAtoms(self, mol):
        mol.typeAtoms()
        assert len([x.GetIntProp('type_id') for x in mol.GetAtoms()]) == 6

    def testBalanceCharge(self, mol):
        mol.typeAtoms()
        mol.balanceCharge()
        np.testing.assert_almost_equal(mol.nbr_charge[3], 0.08, 5)


# class TestDataOne:
#
#     @pytest.fixture
#     def lmp_data(self):
#         mol = rdkitutils.get_mol_from_smiles(CC3COOH)
#         oplsua.OplsTyper(mol).run()
#         ff = oplsua.get_opls_parser()
#         options = get_options()
#         return oplsua.DataOne({1: mol}, ff, 'lmp', options=options)
#
#     def testBalanceCharge(self, lmp_data):
#         assert all([not x.HasProp('neighbor_charge') for x in lmp_data.atom])
#         lmp_data.balanceCharge()
#         charge = [round(x, 2) for x in lmp_data.nbr_charge[1].values()]
#         assert 3 == charge.count(0.08)
#
#     def testSetBonds(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.setBonds()
#         assert 17 == len(lmp_data.bonds)
#
#     def testAdjustBondLength(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.setBonds()
#         lmp_data.adjustBondLength()
#         conf = lmp_data.mols[1].GetConformer()
#         # O-H bond length
#         np.testing.assert_almost_equal(
#             0.945, Chem.rdMolTransforms.GetBondLength(conf, 0, 1))
#
#     def testSetAngles(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.setAngles()
#         assert 21 == len(lmp_data.angles)
#
#     def testSetDihedrals(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.setDihedrals()
#         assert 23 == len(lmp_data.dihedrals)
#
#     def testSetImproperSymbols(self, lmp_data):
#         lmp_data.setImproperSymbols()
#         assert 11 == len(lmp_data.symbol_impropers)
#
#     def testSetImpropers(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.setImproperSymbols()
#         lmp_data.setImpropers()
#         assert 5 == len(lmp_data.impropers)
#
#
# class TestData:
#
#     @pytest.fixture
#     def lmp_data(self):
#         mol1 = rdkitutils.get_mol_from_smiles(BUTANE)
#         mol2 = rdkitutils.get_mol_from_smiles(CC3COOH)
#         mols = {1: mol1, 2: mol2}
#         oplsua.OplsTyper(mol1).run()
#         oplsua.OplsTyper(mol2).run()
#         ff = oplsua.get_opls_parser()
#         options = get_options()
#         return oplsua.Data(mols, ff, 'lmp', options=options)
#
#     def testWriteData(self, lmp_data, tmp_path):
#         with fileutils.chdir(tmp_path, rmtree=True):
#             lmp_data.writeData()
#             assert os.path.exists('lmp.data')
#
#     def testWriteLammpsIn(self, lmp_data, tmp_path):
#         with fileutils.chdir(tmp_path, rmtree=True):
#             lmp_data.writeLammpsIn()
#             assert os.path.exists('lmp.in')
#
#     def testWriteLammpsIn_withData(self, lmp_data, tmp_path):
#         with fileutils.chdir(tmp_path, rmtree=True):
#             lmp_data.writeData()
#             lmp_data.writeLammpsIn()
#             msg = 'fix rigid all shake 0.0001 10 10000 b 3 a 3'
#             matches = [re.search(msg, x) for x in open('lmp.in')]
#             assert any(matches)
#
#     def testSetAtoms(self, lmp_data):
#         lmp_data.setAtoms()
#         assert 22 == len(set([x.GetIntProp('atom_id') for x in lmp_data.atom]))
#
#     def testRemoveAngles(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.setAngles()
#         lmp_data.setImproperSymbols()
#         lmp_data.setImpropers()
#         assert 23 == len(lmp_data.angles)
#         lmp_data.removeAngles()
#         assert 18 == len(lmp_data.angles)
#
#     def testRemoveAngles(self, lmp_data):
#         lmp_data.setAtoms()
#         lmp_data.removeUnused()
#         assert 8 == len(lmp_data.atm_types)
#
#
# class TestDataFileReader:
#
#     @pytest.fixture
#     def raw_dfr(self):
#         return oplsua.DataFileReader(BUTANE_DATA)
#
#     @pytest.fixture
#     def dfr(self):
#         dfr = oplsua.DataFileReader(BUTANE_DATA)
#         dfr.read()
#         dfr.indexLines()
#         dfr.setDescription()
#         return dfr
#
#     @pytest.fixture
#     def df_reader(self):
#         df_reader = oplsua.DataFileReader(BUTANE_DATA)
#         df_reader.run()
#         return df_reader
#
#     def testRead(self, raw_dfr):
#         raw_dfr.read()
#         assert 210 == len(raw_dfr.lines)
#
#     def testIndexLines(self, raw_dfr):
#         raw_dfr.read()
#         raw_dfr.indexLines()
#         assert 11 == len(raw_dfr.mk_idxes)
#
#     def testSetDescription(self, raw_dfr):
#         raw_dfr.read()
#         raw_dfr.indexLines()
#         raw_dfr.setDescription()
#         assert 5 == len(raw_dfr.struct_dsp)
#         assert 5 == len(raw_dfr.dype_dsp)
#         assert 3 == len(raw_dfr.box_dsp)
#
#     def testSetMasses(self, dfr):
#         dfr.setMasses()
#         assert 8 == len(dfr.masses)
#
#     def testSetAtoms(self, dfr):
#         dfr.setMasses()
#         dfr.setAtoms()
#         assert 30 == len(dfr.atoms)
#
#     def testSetMols(self, dfr):
#         dfr.setMasses()
#         dfr.setAtoms()
#         dfr.setMols()
#         assert 3 == len(dfr.mols)
#
#     def testSetBonds(self, dfr):
#         dfr.setBonds()
#         assert 27 == len(dfr.bonds)
#
#     def testAngles(self, dfr):
#         dfr.setAngles()
#         assert 31 == len(dfr.angles)
#
#     def testSetDihedrals(self, dfr):
#         dfr.setDihedrals()
#         assert 30 == len(dfr.dihedrals)
#
#     def testSetImpropers(self, dfr):
#         dfr.setImpropers()
#         assert 7 == len(dfr.impropers)
#
#     @pytest.mark.parametrize(('include14', 'num'), [(True, 9), (False, 5)])
#     def testSetClashExclusion(self, df_reader, include14, num):
#         df_reader.setClashExclusion(include14=include14)
#         assert num == len(df_reader.excluded[7])
#
#     def testSetPairCoeffs(self, dfr):
#         dfr.setPairCoeffs()
#         assert 8 == len(dfr.vdws)
#
#     def testSetVdwRadius(self, df_reader):
#         df_reader.setPairCoeffs()
#         df_reader.setVdwRadius()
#         # gid starts from 1 but there is one placeholder at index 0 for speeding
#         assert 9 == len(df_reader.radii)
