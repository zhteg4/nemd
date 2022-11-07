import math
import copy
import sys
import argparse
import logutils
import functools
import os
import sys

import opls
import units
import parserutils
import fileutils
import nemd
import itertools
import plotutils
import collections
import environutils
import jobutils
import symbols
import numpy as np
import oplsua
from rdkit import Chem
from rdkit.Chem import AllChem

FlAG_CRU = 'cru'
FlAG_CRU_NUM = '-cru_num'
FlAG_MOL_NUM = '-mol_num'

MOLT_OUT_EXT = fileutils.MOLT_FF_EXT

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')

LEN_CH = 1.0930  # length of the C-H bond
# ~= 109.5 degrees = tetrahedronal angle (C-C-C angle)
TETRAHEDRONAL_ANGLE = 2 * math.atan(math.sqrt(2))


def log_debug(msg):
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_error(msg):
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


def get_parser():
    parser = parserutils.get_parser(
        description='Generate the moltemplate input *.lt')
    parser.add_argument(FlAG_CRU,
                        metavar=FlAG_CRU.upper(),
                        type=functools.partial(parserutils.type_monomer_smiles,
                                               allow_mol=True),
                        help='')
    parser.add_argument(FlAG_CRU_NUM,
                        metavar=FlAG_CRU_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        default=1,
                        help='')
    parser.add_argument(FlAG_MOL_NUM,
                        metavar=FlAG_MOL_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        default=1,
                        help='')

    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


class Polymer(object):
    ATOM_ID = oplsua.LammpsWriter.ATOM_ID
    TYPE_ID = oplsua.LammpsWriter.TYPE_ID
    BOND_ATM_ID = oplsua.LammpsWriter.BOND_ATM_ID
    RES_NUM = oplsua.LammpsWriter.RES_NUM
    NEIGHBOR_CHARGE = oplsua.LammpsWriter.NEIGHBOR_CHARGE
    MOL_NUM = 'mol_num'
    IMPLICIT_H = oplsua.LammpsWriter.IMPLICIT_H

    def __init__(self, options, jobname, ff=None):
        self.options = options
        self.jobname = jobname
        self.ff = ff
        self.outfile = self.jobname + MOLT_OUT_EXT
        self.polym = None
        self.polym_Hs = None
        self.mols = {}
        self.buffer = oplsua.LammpsWriter.BUFFER
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()

    def run(self):
        self.setCruMol()
        self.polymerize()
        self.assignAtomType()
        self.balanceCharge()
        self.embedMol()
        self.setMols()
        self.write()
        log('Finished', timestamp=True)

    def setCruMol(self):

        cru_mol = Chem.MolFromSmiles(self.options.cru)
        for atom in cru_mol.GetAtoms():
            if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
                continue
            atom.SetIntProp(self.IMPLICIT_H, atom.GetNumImplicitHs())
            atom.SetNoImplicit(True)
        self.cru_mol = Chem.AddHs(cru_mol)

    def polymerize(self):

        if not symbols.WILD_CARD in self.options.cru:
            # FIXME: We should should polymers and blends with regular molecules
            self.polym = self.cru_mol
            return

        mols = [copy.copy(self.cru_mol) for x in range(self.options.cru_num)]

        combo = mols[0]
        for mol in mols[1:]:
            combo = Chem.CombineMols(combo, mol)
        capping_atoms = [
            x for x in combo.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
        ]
        ht_atoms = [x.GetNeighbors()[0] for x in capping_atoms]
        ht_atom_idxs = [x.GetIdx() for x in ht_atoms]
        edcombo = Chem.EditableMol(combo)
        for t_atom_idx, h_atom_idx in zip(
                ht_atom_idxs[1:-1:2],
                ht_atom_idxs[2::2],
        ):
            edcombo.AddBond(t_atom_idx,
                            h_atom_idx,
                            order=Chem.rdchem.BondType.SINGLE)
        polym = edcombo.GetMol()
        self.polym = Chem.DeleteSubstructs(
            polym, Chem.MolFromSmiles(symbols.WILD_CARD))
        log(f"{Chem.MolToSmiles(self.polym)}")

    def assignAtomType(self):
        res_num = 1
        for sml in self.ff.SMILES:
            if all(x.HasProp(self.TYPE_ID) for x in self.polym.GetAtoms()):
                return
            frag = Chem.MolFromSmiles(sml.sml)
            matches = self.polym.GetSubstructMatches(frag)
            for match in matches:
                frag_cnnt = [
                    x.GetNumImplicitHs() +
                    x.GetDegree() if x.GetSymbol() != 'C' else x.GetDegree()
                    for x in frag.GetAtoms()
                ]
                polm_cnnt = [
                    self.polym.GetAtomWithIdx(x).GetDegree() for x in match
                ]
                match = [
                    x if y == z else None
                    for x, y, z in zip(match, frag_cnnt, polm_cnnt)
                ]
                log_debug(f"assignAtomType {sml.sml}, {match}")
                succeed = self.markAtoms(match, sml, res_num)
                if succeed:
                    res_num += 1

    def markAtoms(self, match, sml, res_num):
        marked = False
        for atom_id, type_id in zip(match, sml.mp):
            if not type_id or atom_id is None:
                continue
            atom = self.polym.GetAtomWithIdx(atom_id)
            try:
                atom.GetIntProp(self.TYPE_ID)
            except KeyError:
                self.setAtomIds(atom, type_id, res_num)
                marked = True
                log_debug(
                    f"{atom.GetSymbol()}{atom.GetDegree()} {atom_id} {type_id}"
                )
            else:
                continue
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    type_id = sml.hs[type_id]
                    self.setAtomIds(neighbor, type_id, res_num)
                    marked = True
                    log_debug(
                        f"{neighbor.GetSymbol()}{neighbor.GetDegree()} {neighbor.GetIdx()} {type_id}"
                    )
        return marked

    def setAtomIds(self, atom, type_id, res_num):
        atom.SetIntProp(self.TYPE_ID, type_id)
        atom.SetIntProp(self.RES_NUM, res_num)
        atom.SetIntProp(self.BOND_ATM_ID,
                        oplsua.OPLS_Parser.BOND_ATOM[type_id])

    def balanceCharge(self):
        res_charge = collections.defaultdict(float)
        for atom in self.polym.GetAtoms():
            res_num = atom.GetIntProp(self.RES_NUM)
            type_id = atom.GetIntProp(self.TYPE_ID)
            res_charge[res_num] += self.ff.charges[type_id]
        for bond in self.polym.GetBonds():
            batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
            bres_num = batom.GetIntProp(self.RES_NUM)
            eres_num = eatom.GetIntProp(self.RES_NUM)
            if bres_num == eres_num:
                continue
            for atom, natom in [[batom, eatom], [eatom, batom]]:
                try:
                    charge = atom.GetDoubleProp(self.NEIGHBOR_CHARGE)
                except KeyError:
                    charge = 0.0
                ncharge = res_charge[natom.GetIntProp(self.RES_NUM)]
                atom.SetDoubleProp(self.NEIGHBOR_CHARGE, charge - ncharge)

    def embedMol(self):
        AllChem.EmbedMolecule(self.polym)

    def setMols(self):

        conformer = self.polym.GetConformer(0)
        xyzs = np.array([
            conformer.GetAtomPosition(x.GetIdx())
            for x in self.polym.GetAtoms()
        ])
        mol_bndr = xyzs.max(axis=0) - xyzs.min(axis=0)
        mol_vec = mol_bndr + self.buffer
        idxs = range(math.ceil(math.pow(self.options.mol_num, 1. / 3)))
        idxs_3d = itertools.product(idxs, idxs, idxs)
        for mol_id, idxs in enumerate(idxs_3d, 1):
            if mol_id > self.options.mol_num:
                return
            polym = copy.copy(self.polym)
            self.mols[mol_id] = polym
            polym.SetIntProp(self.MOL_NUM, mol_id)
            polym_conformer = polym.GetConformer(0)
            for atom in polym.GetAtoms():
                atom_id = atom.GetIdx()
                xyz = list(conformer.GetAtomPosition(atom_id))
                xyz += mol_vec * idxs
                polym_conformer.SetAtomPosition(atom_id, xyz)

    def write(self):
        lmw = oplsua.LammpsWriter(self.ff, self.jobname, mols=self.mols)
        lmw.writeLammpsData()
        lmw.writeLammpsIn()


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    polm = Polymer(options, jobname)
    polm.run()


if __name__ == "__main__":
    main(sys.argv[1:])
