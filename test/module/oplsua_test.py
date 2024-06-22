import os
import pytest
from rdkit import Chem

from nemd import oplsua
from nemd import testutils
from nemd import rdkitutils

BUTANE = 'CCCC'
CC3COOH = '[H]OC(=O)CCC(CC(C)C(=O)O[H])C(=O)O[H]'
BUTANE_DATA = testutils.test_file(os.path.join('polym_builder',
                                               'cooh123.data'))


class TestOplsTyper:

    CCOOH_SML = [
        x for x in oplsua.OplsTyper.SMILES_TEMPLATE if x.sml == 'CC(=O)O'
    ][0]

    @pytest.fixture
    def opls_typer(self):
        mol = rdkitutils.get_mol_from_smiles(CC3COOH, embeded=False)
        return oplsua.OplsTyper(mol)

    def testRun(self, opls_typer):
        opls_typer.run()
        assert all([x.HasProp('type_id') for x in opls_typer.mol.GetAtoms()])

    def testFilterMatch(self, opls_typer):
        frag = Chem.MolFromSmiles(self.CCOOH_SML.sml)
        matches = opls_typer.mol.GetSubstructMatches(frag)
        matches = [opls_typer.filterMatch(x, frag) for x in matches]
        for match in matches:
            assert match[0] is None

    def testMarkMatches(self, opls_typer):
        matches = [[None, 2, 3, 1], [None, 14, 15, 16], [None, 10, 11, 12]]
        res_num, matom_ids = opls_typer.markMatches(matches, self.CCOOH_SML, 1)
        assert 4 == res_num

    def testMarkAtoms(self, opls_typer):
        marked = opls_typer.markAtoms([None, 2, 3, 1], self.CCOOH_SML, 1)
        assert 4 == len(marked)

    def testMarkAtom(self, opls_typer):
        atom = opls_typer.mol.GetAtomWithIdx(2)
        opls_typer.markAtom(atom, 133, 1)
        assert 133 == atom.GetIntProp('type_id')
        assert 1 == atom.GetIntProp('res_num')


class TestOplsParser:

    @pytest.fixture
    def nprsr(self):
        return oplsua.OplsParser()

    @pytest.fixture
    def raw_prsr(self):
        raw_prsr = oplsua.OplsParser()
        raw_prsr.setRawContent()
        raw_prsr.setAtomType()
        return raw_prsr

    def testSetRawContent(self, nprsr):
        nprsr.setRawContent()
        assert 10 == len(nprsr.raw_content)

    def testSetAtomType(self, raw_prsr):
        raw_prsr.setAtomType()
        assert 215 == len(raw_prsr.atoms)

    def testSetVdW(self, raw_prsr):
        raw_prsr.setVdW()
        assert 215 == len(raw_prsr.vdws)

    def testSetCharge(self, raw_prsr):
        raw_prsr.setCharge()
        assert 215 == len(raw_prsr.charges)

    def testSetBond(self, raw_prsr):
        raw_prsr.setBond()
        assert 151 == len(raw_prsr.bonds)

    def testSetAngle(self, raw_prsr):
        raw_prsr.setAngle()
        assert 309 == len(raw_prsr.angles)

    def testSetImproper(self, raw_prsr):
        raw_prsr.setImproper()
        assert 76 == len(raw_prsr.impropers)

    def testSetDihedral(self, raw_prsr):
        raw_prsr.setDihedral()
        assert 630 == len(raw_prsr.dihedrals)

