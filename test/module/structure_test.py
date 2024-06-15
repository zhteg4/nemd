import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from nemd import structure


class TestConformer:

    @pytest.fixture
    def conf(self):
        mol = structure.Mol(Chem.MolFromSmiles('CCCCC'))
        conf = structure.Conformer(mol.GetNumAtoms())
        return mol.AddConformer(conf)

    def testSetGids(self, conf):
        conf.setGids(2)
        np.testing.assert_array_equal(conf.id_map, [2, 3, 4, 5, 6])
        assert conf.id_map[1] == 3

    def testAids(self, conf):
        conf.setGids(1)
        np.testing.assert_array_equal(conf.aids, [0, 1, 2, 3, 4])

    def testSetPositions(self, conf):
        xyz = np.zeros(conf.GetPositions().shape) + 1
        conf.setPositions(xyz)
        np.testing.assert_array_equal(conf.GetPositions(), xyz)

    def testCentroid(self, conf):
        assert np.average(conf.centroid()) == 0

    def testTranslate(self, conf):
        conf.translate([1, 2, 3])
        np.testing.assert_array_equal(conf.centroid(), [1, 2, 3])

    def testSetBondLength(self, conf):
        xyz = np.array([x * 0.1 for x in range(15)]).reshape(-1, 3)
        conf.setPositions(xyz)
        conf.setBondLength((0, 1), 2)
        np.testing.assert_almost_equal(
            Chem.rdMolTransforms.GetBondLength(conf, 0, 1), 2)


class TestMol:
    pass

    # def testAddConformer(self, conf):
    #     mol = structure.Mol(Chem.MolFromSmiles('CCCCO'))
    #     assert conf.GetOwningMol() != mol
    #     mol.AddConformer(conf)
    #     assert conf.GetOwningMol() == mol

    # def testSetOwningMol(self, conf):
    #     mol = structure.Mol(Chem.MolFromSmiles('CCCCO'))
    #     conf.setOwningMol(mol)
    #     assert conf.GetOwningMol() == mol
