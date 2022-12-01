import os
import pytest
import fragments
import testutils
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from contextlib import contextmanager

BUTANE = 'CCCC'
BUTENE = 'CC=CC'
CCCOOH = '[H]OC(=O)CC'
BENZENE = 'C1=CC=CC=C1'
CC3COOH = '[H]OC(=O)CCC(CC(C)C(=O)O[H])C(=O)O[H]'
COOHPOLYM = 'This is the above SMILES with polymer properties marked'
POLYM_BUILDER = 'polym_builder'
BASE_DIR = testutils.test_file(POLYM_BUILDER)
BUTANE_DATA = os.path.join(BASE_DIR, 'butane.data')


def getMol(smiles_str):
    real_smiles_str = CC3COOH if smiles_str == COOHPOLYM else smiles_str
    with rdkit_preserve_hs() as ps:
        mol = Chem.MolFromSmiles(real_smiles_str, ps)
    with rdkit_warnings_ignored():
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    if smiles_str == COOHPOLYM:
        markPolymProps(mol)
    return mol


def markPolymProps(mol):
    m1_ids = [0, 1, 2, 3, 4, 5]
    m2_ids = [17, 16, 14, 15, 6, 7]
    m3_ids = [13, 12, 10, 11, 8, 9]
    for mono_id, ids in enumerate([m1_ids, m2_ids, m3_ids]):
        for id in ids:
            atom = mol.GetAtomWithIdx(id)
            atom.SetIntProp(fragments.FragMol.MONO_ID, mono_id)
    for ht_atom_id in [4, 9]:
        atom = mol.GetAtomWithIdx(ht_atom_id)
        atom.SetBoolProp(fragments.FragMol.POLYM_HT, True)
    mol.SetBoolProp(fragments.FragMol.IS_MONO, True)
    return mol


@contextmanager
def rdkit_preserve_hs():
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    try:
        yield ps
    finally:
        ps.removeHs = True


@contextmanager
def rdkit_warnings_ignored():
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    try:
        yield lg
    finally:
        lg.setLevel(RDLogger.WARNING)


class TestFragMol:

    @pytest.fixture
    def fmol(self, smiles_str, data_file):
        mol = getMol(smiles_str)
        return fragments.FragMol(mol, data_file)

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'rotatable'),
                             [(BUTANE, None, True), (BUTENE, None, False),
                              (CCCOOH, None, True), (BENZENE, None, False)])
    def testIsRotatable(self, fmol, rotatable):
        assert rotatable == fmol.isRotatable([1, 2])

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'num'),
                             [(BUTANE, None, 1), (CCCOOH, None, 3),
                              (CC3COOH, None, 15)])
    def testGetSwingAtoms(self, fmol, num):
        assert num == len(fmol.getSwingAtoms(*[0, 1, 2, 3]))

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'src', 'trgt', 'pln'),
                             [(BUTANE, None, 0, 3, 4), (CCCOOH, None, 0, 5, 5),
                              (CC3COOH, None, 0, 13, 11),
                              (COOHPOLYM, None, 0, 13, 11)])
    def testfindPath(self, fmol, src, trgt, pln):
        nsrc, ntrgt, npath = fmol.findPath()
        assert src == nsrc
        assert trgt == ntrgt
        assert pln == len(npath)

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'num'),
                             [(BUTANE, None, 1), (BUTENE, None, 0),
                              (CCCOOH, None, 2), (BENZENE, None, 0),
                              (COOHPOLYM, None, 10)])
    def testAddNxtFrags(self, fmol, num):
        fmol.addNxtFrags()
        assert num == fmol.getNumFrags()

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'has_pre'),
                             [(BUTANE, None, False), (BUTENE, None, None),
                              (CCCOOH, None, True), (BENZENE, None, None),
                              (COOHPOLYM, None, True)])
    def testSetPreFrags(self, fmol, has_pre):
        fmol.addNxtFrags()
        fmol.setPreFrags()
        frags = fmol.fragments()
        if not frags:
            return
        assert has_pre == bool(frags[-1].pfrag)

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'num'),
                             [(BUTANE, BUTANE_DATA, 4)])
    def testReadData(self, fmol, num):
        fmol.readData()
        assert num == len(fmol.data_reader.radii)

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetDCellParams(self, fmol):
        fmol.readData()
        fmol.setDCellParams()
        np.testing.assert_allclose(2.629928579188861, fmol.cell_rez)
        np.testing.assert_allclose(5.259857158377722, fmol.cell_cut)

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetCoords(self, fmol):
        fmol.readData()
        fmol.setCoords()
        dihe = Chem.rdMolTransforms.GetDihedralDeg(fmol.conf, 0, 1, 2, 3)
        np.testing.assert_allclose(54.70031111417669, dihe)

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetFrm(self, fmol):
        fmol.readData()
        fmol.setCoords()
        fmol.setFrm()
        assert (4, 3) == fmol.frm.shape

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetDcell(self, fmol):
        fmol.readData()
        fmol.setDCellParams()
        fmol.setCoords()
        fmol.setFrm()
        fmol.setDcell()
        assert 3 == fmol.dcell.grids.shape[0]

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testHasClashes(self, fmol):
        fmol.addNxtFrags()
        fmol.setPreFrags()
        fmol.setInitAtomIds()
        fmol.readData()
        fmol.setDCellParams()
        fmol.setCoords()
        fmol.setFrm()
        fmol.setDcell()
        assert not fmol.hasClashes([3])
        fmol.data_reader.radii[4][1] = 3
        assert fmol.hasClashes([3])

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetConformer(self, fmol):
        fmol.addNxtFrags()
        fmol.setPreFrags()
        fmol.setInitAtomIds()
        fmol.readData()
        fmol.setDCellParams()
        fmol.setCoords()
        fmol.setFrm()
        fmol.setDcell()
        fmol.data_reader.radii[4][1] = 3
        assert fmol.hasClashes([3])
        fmol.setConformer()
        assert not fmol.hasClashes([3])
