import types
import pytest
import numpy as np

from nemd import lammpsdata


class TestConformer:

    @pytest.fixture
    def conf(self):
        mol = lammpsdata.Mol.MolFromSmiles('CCCC(C)C')
        mol.EmbedMolecule()
        return mol.GetConformer()

    def testAtoms(self, conf):
        assert conf.atoms.shape == (6, 6)

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
    def raw_mol(self):
        mol = lammpsdata.Mol.MolFromSmiles('[H]OC(=O)CC', delay=True)
        mol.EmbedMolecule()
        return mol

    def testTypeAtoms(self, raw_mol):
        raw_mol.typeAtoms()
        assert len([x.GetIntProp('type_id') for x in raw_mol.GetAtoms()]) == 6

    def testBalanceCharge(self, raw_mol):
        raw_mol.typeAtoms()
        raw_mol.balanceCharge()
        np.testing.assert_almost_equal(raw_mol.nbr_charge[1], 0.08, 5)

    @pytest.fixture
    def mol(self):
        mol = lammpsdata.Mol.MolFromSmiles('[H]OC(=O)CC')
        mol.EmbedMolecule()
        return mol

    def testAtoms(self, mol):
        assert mol.atoms.shape == (6, 6)

    def testBonds(self, mol):
        assert mol.bonds.shape == (5, 3)

    def testAngles(self, mol):
        assert mol.angles.shape == (4, 4)

    def testDihedrals(self, mol):
        assert mol.dihedrals.shape == (4, 5)

    def testImpropers(self, mol):
        assert mol.impropers.shape == (1, 5)

    def testSetFixGeom(self, mol):
        bonds, angles = mol.getRigid()
        assert all(bonds == 86)
        assert all(angles == 296)

    def testMolecularWeight(self, mol):
        assert mol.mw == 74.079


class TestBase:

    @pytest.fixture
    def base(self):
        return lammpsdata.Base(options=types.SimpleNamespace(jobname='test'))

    def testHeader(self, base):
        assert base.header == ''
