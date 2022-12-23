import os
import pytest
import oplsua
import fileutils
import testutils
import rdkitutils
from rdkit import Chem

BUTANE = 'CCCC'
CC3COOH = '[H]OC(=O)CCC(CC(C)C(=O)O[H])C(=O)O[H]'
BUTANE_DATA = testutils.test_file(os.path.join('polym_builder',
                                               'cooh123.data'))


class TestOplsTyper:

    CCOOH_SML = [x for x in oplsua.OplsTyper.SMILES if x.sml == 'CC(=O)O'][0]

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


class TestLammpsIn:

    @pytest.fixture
    def lmp_in(self):
        return oplsua.LammpsIn('lmp')

    def testWriteLammpsIn(self, lmp_in, tmp_path):
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_in.writeLammpsIn()
            assert os.path.exists('lmp.in')


class TestLammpsWriter:

    @pytest.fixture
    def lmp_data(self):
        mol1 = rdkitutils.get_mol_from_smiles(BUTANE)
        mol2 = rdkitutils.get_mol_from_smiles(CC3COOH)
        oplsua.OplsTyper(mol1).run()
        oplsua.OplsTyper(mol2).run()
        ff = oplsua.get_opls_parser()
        return oplsua.LammpsWriter({1: mol1, 2: mol2}, ff, 'lmp')

    def testWriteData(self, lmp_data, tmp_path):
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_data.writeData()
            assert os.path.exists('lmp.data')

    def testWriteLammpsIn(self, lmp_data, tmp_path):
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_data.writeLammpsIn()
            assert os.path.exists('lmp.in')

    def testSetAtoms(self, lmp_data):
        lmp_data.setAtoms()
        assert 22 == len(set([x.GetIntProp('atom_id') for x in lmp_data.atom]))

    def testBalanceCharge(self, lmp_data):
        assert all([not x.HasProp('neighbor_charge') for x in lmp_data.atom])
        lmp_data.balanceCharge()
        atoms = [x for x in lmp_data.atom if x.HasProp('neighbor_charge')]
        nchrgs = [round(x.GetDoubleProp('neighbor_charge'), 4) for x in atoms]
        assert 3 == nchrgs.count(0.08)

    def testSetBonds(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setBonds()
        assert 20 == len(lmp_data.bonds)

    def testAdjustBondLength(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setBonds()
        lmp_data.adjustBondLength()
        conf = lmp_data.mols[1].GetConformer()
        assert 1.526 == Chem.rdMolTransforms.GetBondLength(conf, 0, 1)

    def testSetAngles(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setAngles()
        assert 23 == len(lmp_data.angles)

    def testSetDihedrals(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setDihedrals()
        assert 24 == len(lmp_data.dihedrals)

    def testSetImproperSymbols(self, lmp_data):
        lmp_data.setImproperSymbols()
        assert 10 == len(lmp_data.symbol_impropers)

    def testSetImpropers(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setImproperSymbols()
        lmp_data.setImpropers()
        assert 5 == len(lmp_data.impropers)

    def testRemoveAngles(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setAngles()
        lmp_data.setImproperSymbols()
        lmp_data.setImpropers()
        assert 23 == len(lmp_data.angles)
        lmp_data.removeAngles()
        assert 18 == len(lmp_data.angles)

    def testRemoveAngles(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.removeUnused()
        assert 8 == len(lmp_data.atm_types)


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
        assert 30 == len(df_reader.radii)
