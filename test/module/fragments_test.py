import os
import pytest
import numpy as np
from rdkit import Chem
from nemd import fragments
from nemd import testutils
from nemd import rdkitutils

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
    mol = rdkitutils.get_mol_from_smiles(real_smiles_str)
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
                             [(BUTANE, None, 1), (BUTENE, None, 1),
                              (CCCOOH, None, 2), (BENZENE, None, 1),
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
        if not frags[0].dihe:
            return
        assert has_pre == bool(frags[-1].pfrag)

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'num'),
                             [(BUTANE, BUTANE_DATA, 3)])
    def testReadData(self, fmol, num):
        fmol.readData()
        assert num == len(fmol.data_reader.radii)

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetDCellParams(self, fmol):
        fmol.readData()
        fmol.setDCellParams()
        np.testing.assert_allclose(1.97168, fmol.cell_cut, 0.001)

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
        fmol.readData(include14=True)
        fmol.setDCellParams()
        fmol.setCoords()
        fmol.setFrm()
        fmol.setDcell()
        fmol.data_reader.radii.setRadius(1, 4, 4)
        assert fmol.hasClashes([3])
        fmol.data_reader.radii.setRadius(1, 4, 2)
        assert not fmol.hasClashes([3])

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testSetConformer(self, fmol):
        fmol.addNxtFrags()
        fmol.setPreFrags()
        fmol.setInitAtomIds()
        fmol.readData(include14=True)
        fmol.setDCellParams()
        fmol.setCoords()
        fmol.setFrm()
        fmol.setDcell()
        fmol.data_reader.radii[4][1] = 3
        assert fmol.hasClashes([3])
        fmol.setConformer()
        assert not fmol.hasClashes([3])


class TestFragment:

    @pytest.fixture
    def frag(self, smiles_str, data_file):
        mol = getMol(smiles_str)
        fmol = fragments.FragMol(mol, data_file)
        frag = fragments.Fragment([], fmol)
        return frag

    @pytest.mark.parametrize(('smiles_str', 'data_file'), [(BUTANE, None)])
    def testResetVals(self, frag):
        frag.resetVals()
        assert not frag.val
        assert frag.fval

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'num'),
                             [(BUTANE, None, 1), (BUTENE, None, 0),
                              (CCCOOH, None, 2), (BENZENE, None, 0),
                              (CC3COOH, None, 8)])
    def testSetFrags(self, frag, num):
        assert num == len(frag.setFrags())

    @pytest.mark.parametrize(('smiles_str', 'data_file', 'num'),
                             [(BUTANE, None, 1), (BUTENE, None, 0),
                              (CCCOOH, None, 2), (BENZENE, None, 0),
                              (CC3COOH, None, 8)])
    def testGetNewDihes(self, frag, num):
        assert num == len(frag.getNewDihes())

    @pytest.mark.parametrize(('smiles_str', 'data_file'), [(BUTANE, None),
                                                           (CCCOOH, None)])
    def testSetDihedralDeg(self, frag):
        frag.setFrags()
        frag.setDihedralDeg(123)
        assert np.isclose(123, frag.getDihedralDeg())
        assert frag.fval
        frag.setDihedralDeg()
        assert not np.isclose(123, frag.getDihedralDeg())
        assert not frag.fval

    @pytest.mark.parametrize(('smiles_str', 'data_file'), [(BUTANE, None),
                                                           (BUTENE, None),
                                                           (CCCOOH, None),
                                                           (BENZENE, None)])
    def testPopVal(self, frag):
        num_vals = len(frag.vals)
        frag.val = frag.popVal()
        assert frag.val is not None
        assert num_vals != len(frag.vals)
        assert not frag.fval

    @pytest.mark.parametrize(('smiles_str', 'data_file'), [(CC3COOH, None)])
    def testGetPreAvailFrag(self, frag):
        fmol = frag.fmol
        fmol.addNxtFrags()
        fmol.setPreFrags()
        # frag with no previous
        assert fmol.init_frag.getPreAvailFrag() is None
        # frag without dihedral vals return one available previous
        second_frag = fmol.init_frag.nfrags[0]
        second_frag.vals = []
        assert 0 == second_frag.getPreAvailFrag().dihe[0]
        # No available previous
        fmol.init_frag.vals = []
        assert second_frag.getPreAvailFrag() is None

    @pytest.mark.parametrize(('smiles_str', 'data_file'), [(CC3COOH, None)])
    def testGetNxtFrags(self, frag):
        fmol = frag.fmol
        fmol.addNxtFrags()
        fmol.setPreFrags()
        second_frag = fmol.init_frag.nfrags[0]
        second_frag.vals = []
        frag = second_frag.getPreAvailFrag()
        assert 0 == frag.dihe[0]
        frag.getNxtFrags()
        assert not frag.getNxtFrags()
        frag.nfrags[0].fval = False
        assert 1 == len(frag.getNxtFrags())

    @pytest.mark.parametrize(('smiles_str', 'data_file'),
                             [(BUTANE, BUTANE_DATA)])
    def testGetNxtFrags(self, frag):
        fmol = frag.fmol
        fmol.run()
        frag = fmol.init_frag
        oval = frag.getDihedralDeg()
        frag.setConformer()
        assert oval != frag.getDihedralDeg()


class TestFragMols:
    SMILES1 = '[C:1][C:2]([C:3](=[O:4])[O:5][H:6])[C:7][C:8]([C:9](=[O:10])[O:11][H:12])[C:13][C:14]([C:15](=[O:16])[O:17][H:18])[C:19][C:20]([C:21](=[O:22])[O:23][H:24])[C:25][C:26]([C:27](=[O:28])[O:29][H:30])[C:31][C:32][C:33](=[O:34])[O:35][H:36]'
    SMILES2 = '[C:37][C:38]([C:39](=[O:40])[O:41][H:42])[C:43][C:44]([C:45](=[O:46])[O:47][H:48])[C:49][C:50]([C:51](=[O:52])[O:53][H:54])[C:55][C:56]([C:57](=[O:58])[O:59][H:60])[C:61][C:62]([C:63](=[O:64])[O:65][H:66])[C:67][C:68][C:69](=[O:70])[O:71][H:72]'
    SMILES3 = '[C:73][C:74]([C:75](=[O:76])[O:77][H:78])[C:79][C:80]([C:81](=[O:82])[O:83][H:84])[C:85][C:86]([C:87](=[O:88])[O:89][H:90])[C:91][C:92]([C:93](=[O:94])[O:95][H:96])[C:97][C:98]([C:99](=[O:100])[O:101][H:102])[C:103][C:104][C:105](=[O:106])[O:107][H:108]'

    @pytest.fixture
    def fmols(self):
        mol1 = getMol(self.SMILES1)
        mol2 = getMol(self.SMILES2)
        mol3 = getMol(self.SMILES3)
        data_file = testutils.test_file(
            os.path.join('polym_builder', 'cooh6x3.data'))
        box = [
            0, 19.321654197203486, 0, 19.321654197203486, 0, 19.321654197203486
        ]
        fmols = fragments.FragMols({
            1: mol1,
            2: mol2,
            3: mol3
        },
                                   data_file,
                                   box=box)
        return fmols

    def testFragmentize(self, fmols):
        fmols.fragmentize()
        str_fmols_1st = str(fmols.fmols[1].fragments())
        assert str_fmols_1st == str(fmols.fmols[2].fragments())
        assert str_fmols_1st == str(fmols.fmols[2].fragments())
