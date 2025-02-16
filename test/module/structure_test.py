import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms, AllChem

from nemd import structure
from nemd import rdkitutils

MOL_ONLY = Chem.MolFromSmiles('CCCCC')
MOL_WITH_CONF = Chem.MolFromSmiles('CCCCC')
with rdkitutils.rdkit_warnings_ignored():
    Chem.AllChem.EmbedMolecule(MOL_WITH_CONF)
MOL_WITH_CONFS = Chem.MolFromSmiles('CCCCC')
with rdkitutils.rdkit_warnings_ignored():
    Chem.AllChem.EmbedMolecule(MOL_WITH_CONFS)
MOL_WITH_CONFS.AddConformer(MOL_WITH_CONFS.GetConformer(0), assignId=True)


class TestConformer:

    @pytest.fixture
    def conf(self):
        mol = structure.Mol(MOL_ONLY)
        conf = structure.Conformer(mol.GetNumAtoms())
        return mol.AddConformer(conf)

    def testSetUp(self, conf):
        conf.setUp(conf.GetOwningMol(), cid=1, gid=2)
        np.testing.assert_array_equal(conf.id_map, [2, 3, 4, 5, 6])
        assert conf.id_map[1] == 3

    def testAids(self, conf):
        conf.setUp(conf.GetOwningMol(), cid=1, gid=1)
        np.testing.assert_array_equal(conf.aids, [0, 1, 2, 3, 4])

    def testGids(self, conf):
        conf.setUp(conf.GetOwningMol(), cid=1, gid=1)
        np.testing.assert_array_equal(conf.gids, [1, 2, 3, 4, 5])

    def testHasOwningMol(self, conf):
        assert conf.HasOwningMol()
        assert not structure.Conformer().HasOwningMol()

    def testGetOwningMol(self, conf):
        assert conf.GetOwningMol().GetNumAtoms() == 5

    def testSetPositions(self, conf):
        xyz = np.zeros(conf.GetPositions().shape) + 1
        conf.setPositions(xyz)
        np.testing.assert_array_equal(conf.GetPositions(), xyz)


class TestMol:

    STRUCT_WITH_MOL = structure.Struct()
    STRUCT_WITH_MOL.addMol(MOL_WITH_CONFS)

    @pytest.mark.parametrize(
        "imol, struct, num_conf, gids, mgid",
        [(MOL_ONLY, None, 0, [], None), (MOL_WITH_CONF, None, 1, [1], 5),
         (MOL_WITH_CONF, structure.Struct(), 1, [1], 5),
         (MOL_WITH_CONFS, STRUCT_WITH_MOL, 2, [3, 4], 20)])
    def testSetUp(self, imol, struct, num_conf, gids, mgid):
        mol = structure.Mol(imol, struct=struct, delay=True)
        assert not mol.confs
        mol.setUp(imol.GetConformers())
        assert mol.GetNumConformers() == num_conf
        assert [x.gid for x in mol.GetConformers()] == gids
        if mgid:
            assert max([x.id_map.max() for x in mol.GetConformers()]) == mgid

    @pytest.fixture
    def mol(self):
        return structure.Mol(MOL_WITH_CONFS, struct=self.STRUCT_WITH_MOL)

    def testGetConformer(self, mol):
        assert mol.GetConformer(1).GetId() == 1
        assert mol.GetConformer(0).GetId() == 0

    def testAddConformer(self, mol):
        conf = mol.AddConformer(structure.Conformer(mol.GetNumAtoms()))
        assert mol.GetNumConformers() == 3
        assert conf.gid == 3
        assert conf.id_map.max() == 15

    def testGetConformers(self, mol):
        assert len(mol.GetConformers()) == 2

    def testEmbedMolecule(self, mol):
        mol.EmbedMolecule()
        assert mol.GetNumConformers() == 1
        mol.AddConformer(mol.GetConformer(0))
        assert mol.GetNumConformers() == 2
        mol.EmbedMolecule()
        assert mol.GetNumConformers() == 1

    def testMolecularWeight(self, mol):
        assert mol.mw == 72.093900384

    @pytest.mark.parametrize(
        'smiles,united,out',
        [('CC([H])([H])[H]', True, 'CC'),
         ('CC([H])([H])[H]', False, '[H]C([H])([H])C([H])([H])[H]'),
         ('OC(=O)CC', True, '[H]OC(=O)CC'),
         ('OC(=O)CC', False, '[H]OC(=O)C([H])([H])C([H])([H])[H]')])
    def testMolFromSmiles(self, smiles, united, out):
        assert Chem.MolToSmiles(
            structure.Mol.MolFromSmiles(smiles, united=united)) == out

    def testAtomTotal(self, mol):
        assert mol.atom_total == 10


class TestStruct:

    @pytest.fixture
    def struct(self):
        return structure.Struct.fromMols([MOL_WITH_CONFS, MOL_ONLY])

    def testFromMols(self):
        struct = structure.Struct.fromMols([MOL_WITH_CONFS, MOL_ONLY])
        assert len(struct.molecules) == 2

    def testAddMol(self, struct):
        struct = structure.Struct()
        struct.addMol(MOL_WITH_CONFS)
        assert len(struct.molecules) == 1
        struct.addMol(MOL_ONLY)
        assert len(struct.molecules) == 2

    def testGetIds(self, struct):
        assert struct.getIds() == (3, 11)

    def testConformers(self, struct):
        assert struct.conformer_total == 2

    def testAtoms(self, struct):
        assert len([x for x in struct.atom]) == 10

    def testAtomTotal(self, struct):
        assert struct.atom_total == 10

    def testGetPositions(self, struct):
        assert struct.getPositions().shape == (10, 3)

    def testConformerTotal(self, struct):
        assert struct.conformer_total == 2
