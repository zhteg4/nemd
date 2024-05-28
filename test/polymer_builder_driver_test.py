import pytest
from nemd import testutils

import polymer_builder_driver as driver

BASE_DIR = testutils.test_file('polym_builder')


class TestMol(object):

    @pytest.fixture
    def mol(self):
        parser = driver.get_parser()
        options = parser.parse_args(
            ['C(*)C(*)CO', '-cru_num', '3', '-mol_num', '2'])
        return driver.Mol(options.cru[0],
                          options.cru_num[0],
                          options.mol_num[0],
                          options=options,
                          delay=True)

    def testSetCruMol(self, mol):
        mol.setCruMol()
        assert mol.cru_mol.GetNumAtoms() == 7

    def testMarkMonomer(self, mol):
        mol.setCruMol()
        mol.markMonomer()
        assert mol.cru_mol.GetBoolProp('is_mono')
        ma = [x.GetIntProp('mono_atom_idx') for x in mol.cru_mol.GetAtoms()]
        assert ma == [0, 1, 2, 3, 4, 5, 6]
        assert mol.cru_mol.GetAtomWithIdx(0).GetSymbol() == '*'
        assert mol.cru_mol.GetAtomWithIdx(0).GetBoolProp('cap')
        assert mol.cru_mol.GetAtomWithIdx(1).GetBoolProp('ht')

    def testPolymerize(self, mol):
        mol.setCruMol()
        mol.markMonomer()
        mol.polymerize()
        assert mol.GetNumAtoms() == 15

    def testAssignAtomType(self, mol):
        mol.setCruMol()
        mol.markMonomer()
        mol.polymerize()
        mol.assignAtomType()
        atoms = mol.GetAtoms()
        assert len([x.GetIntProp('type_id') for x in atoms]) == 15
        assert len([x.GetIntProp('res_num') for x in atoms]) == 15

    def testEmbedMol(self, mol):
        mol.setCruMol()
        mol.markMonomer()
        mol.polymerize()
        mol.assignAtomType()
        mol.embedMol()
        assert mol.GetNumConformers() == 1

    def testSetConformers(self, mol):
        mol.setCruMol()
        mol.markMonomer()
        mol.polymerize()
        mol.assignAtomType()
        mol.embedMol()
        mol.setConformers()
        assert mol.GetNumConformers() == 2


# class TestTransConformer(object):
#
#     @pytest.fixture
#     def raw_conf(self):
#         polym = driver.Conformer.read(os.path.join(BASE_DIR, 'polym.sdf'))
#         original_cru_mol = driver.Conformer.read(
#             os.path.join(BASE_DIR, 'original_cru_mol.sdf'))
#         raw_conf = driver.Conformer(polym, original_cru_mol)
#         raw_conf.relax_dir = os.path.join(BASE_DIR, raw_conf.relax_dir)
#         return raw_conf
#
#     def testFoldPolym(self, raw_conf):
#         raw_conf.setCruMol()
#         raw_conf.setCruBackbone()
#         raw_conf.foldPolym()
