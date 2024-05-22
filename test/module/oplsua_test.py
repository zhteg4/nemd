import os
import re
import pytest
import numpy as np
from rdkit import Chem

from nemd import oplsua
from nemd import fileutils
from nemd import testutils
from nemd import rdkitutils
from nemd import parserutils

BUTANE = 'CCCC'
CC3COOH = '[H]OC(=O)CCC(CC(C)C(=O)O[H])C(=O)O[H]'
BUTANE_DATA = testutils.test_file(os.path.join('polym_builder',
                                               'cooh123.data'))


def get_options(args=None):
    if args is None:
        args = []
    parser = parserutils.get_parser()
    parserutils.add_md_arguments(parser)
    return parser.parse_args(args)


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


class TestLammpsIn:

    @pytest.fixture
    def lmp_in(self):
        options = get_options()
        return oplsua.LammpsIn('lmp', options=options)

    def testWriteLammpsIn(self, lmp_in, tmp_path):
        # Write LAMMPS in script without molecule structure known
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_in.writeLammpsIn()
            assert os.path.exists('lmp.in')


class TestLammpsDataOne:

    @pytest.fixture
    def lmp_data(self):
        mol = rdkitutils.get_mol_from_smiles(CC3COOH)
        oplsua.OplsTyper(mol).run()
        ff = oplsua.get_opls_parser()
        options = get_options()
        return oplsua.LammpsDataOne({1: mol}, ff, 'lmp', options=options)

    def testBalanceCharge(self, lmp_data):
        assert all([not x.HasProp('neighbor_charge') for x in lmp_data.atom])
        lmp_data.balanceCharge()
        charge = [round(x, 2) for x in lmp_data.nbr_charge[1].values()]
        assert 3 == charge.count(0.08)

    def testSetBonds(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setBonds()
        assert 17 == len(lmp_data.bonds)

    def testAdjustBondLength(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setBonds()
        lmp_data.adjustBondLength()
        conf = lmp_data.mols[1].GetConformer()
        # O-H bond length
        np.testing.assert_almost_equal(
            0.945, Chem.rdMolTransforms.GetBondLength(conf, 0, 1))

    def testSetAngles(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setAngles()
        assert 21 == len(lmp_data.angles)

    def testSetDihedrals(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setDihedrals()
        assert 23 == len(lmp_data.dihedrals)

    def testSetImproperSymbols(self, lmp_data):
        lmp_data.setImproperSymbols()
        assert 11 == len(lmp_data.symbol_impropers)

    def testSetImpropers(self, lmp_data):
        lmp_data.setAtoms()
        lmp_data.setImproperSymbols()
        lmp_data.setImpropers()
        assert 5 == len(lmp_data.impropers)


class TestLammpsData:

    @pytest.fixture
    def lmp_data(self):
        mol1 = rdkitutils.get_mol_from_smiles(BUTANE)
        mol2 = rdkitutils.get_mol_from_smiles(CC3COOH)
        mols = {1: mol1, 2: mol2}
        oplsua.OplsTyper(mol1).run()
        oplsua.OplsTyper(mol2).run()
        ff = oplsua.get_opls_parser()
        options = get_options()
        return oplsua.LammpsData(mols, ff, 'lmp', options=options)

    def testWriteData(self, lmp_data, tmp_path):
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_data.writeData()
            assert os.path.exists('lmp.data')

    def testWriteLammpsIn(self, lmp_data, tmp_path):
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_data.writeLammpsIn()
            assert os.path.exists('lmp.in')

    def testWriteLammpsIn_withData(self, lmp_data, tmp_path):
        with fileutils.chdir(tmp_path, rmtree=True):
            lmp_data.writeData()
            lmp_data.writeLammpsIn()
            msg = 'fix rigid all shake 0.0001 10 10000 b 3 a 3'
            matches = [re.search(msg, x) for x in open('lmp.in')]
            assert any(matches)

    def testSetAtoms(self, lmp_data):
        lmp_data.setAtoms()
        assert 22 == len(set([x.GetIntProp('atom_id') for x in lmp_data.atom]))

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
        dfr.indexLines()
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

    def testIndexLines(self, raw_dfr):
        raw_dfr.read()
        raw_dfr.indexLines()
        assert 11 == len(raw_dfr.mk_idxes)

    def testSetDescription(self, raw_dfr):
        raw_dfr.read()
        raw_dfr.indexLines()
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
        # gid starts from 1 but there is one placeholder at index 0 for speeding
        assert 9 == len(df_reader.radii)
