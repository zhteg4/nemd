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
        mol = lammpsdata.Mol.MolFromSmiles('[H]OC(=O)CC', delay=True)
        mol.EmbedMolecule()
        return mol

    def testTypeAtoms(self, mol):
        mol.typeAtoms()
        assert len([x.GetIntProp('type_id') for x in mol.GetAtoms()]) == 6

    def testBalanceCharge(self, mol):
        mol.typeAtoms()
        mol.balanceCharge()
        np.testing.assert_almost_equal(mol.nbr_charge[3], 0.08, 5)

    def testSetBonds(self, mol):
        mol.typeAtoms()
        mol.setBonds()
        assert mol.bond_total == 5

    def testSetAngles(self, mol):
        mol.typeAtoms()
        mol.setAngles()
        assert mol.angle_total == 5

    def testSetDihedrals(self, mol):
        mol.typeAtoms()
        mol.setDihedrals()
        assert mol.dihedral_total == 4

    def testSetImpropers(self, mol):
        mol.typeAtoms()
        mol.setImpropers()
        assert mol.improper_total == 1

    def testRemoveAngles(self, mol):
        mol.typeAtoms()
        mol.setAngles()
        mol.setImpropers()
        mol.removeAngles()
        assert len(mol.angles) == 4

    def testSetFixGeom(self, mol):
        mol.typeAtoms()
        mol.setBonds()
        mol.setAngles()
        mol.setFixGeom()
        assert len(mol.fbonds) == 1
        assert len(mol.fangles) == 1

    def testMolecularWeight(self, mol):
        mol.typeAtoms()
        assert mol.mw == 74.079


class TestBase:

    @pytest.fixture
    def base(self):
        return lammpsdata.Base()

    def testHeader(self, base):
        assert base.header == ''
