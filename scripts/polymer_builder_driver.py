import math
import copy
import sys
import argparse
import random
import networkx as nx
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
from rdkit import Geometry

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
                        nargs='+',
                        help='')
    parser.add_argument(FlAG_CRU_NUM,
                        metavar=FlAG_CRU_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        nargs='+',
                        help='')
    parser.add_argument(FlAG_MOL_NUM,
                        metavar=FlAG_MOL_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        nargs='+',
                        help='')

    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


class AmorphousCell(object):

    def __init__(self, options, jobname, ff=None):
        self.options = options
        self.jobname = jobname
        self.outfile = self.jobname + MOLT_OUT_EXT
        self.polymers = []
        self.molecules = {}
        self.mols = {}
        self.ff = ff
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()

    def run(self):
        self.setPolymers()
        self.setMolecules()
        self.write()

    def setPolymers(self):
        for cru, cru_num, in zip(self.options.cru, self.options.cru_num):
            polym = Polymer(cru, cru_num)
            polym.run()
            self.polymers.append(polym)

    def setMolecules(self):
        boxes = []
        for polymer in self.polymers:
            polym = polymer.polym
            conformer = polym.GetConformer(0)
            xyzs = np.array([
                conformer.GetAtomPosition(x.GetIdx())
                for x in polym.GetAtoms()
            ])
            box = xyzs.max(axis=0) - xyzs.min(axis=0) + polymer.buffer
            boxes.append(box)

        boxes = np.array(boxes).reshape(-1, 3)
        mbox = boxes.max(axis=0)
        for polymer, box, mol_num in zip(self.polymers, boxes,
                                         self.options.mol_num):
            polymer.mol_nums_per_mbox = np.floor(mbox / box).astype(int)
            polymer.mol_num_per_mbox = np.product(polymer.mol_nums_per_mbox)
            polymer.mol_num = mol_num
            polymer.num_mbox = math.ceil(polymer.mol_num /
                                         polymer.mol_num_per_mbox)
            percent = [
                np.linspace(-0.5, 0.5, x, endpoint=False)
                for x in polymer.mol_nums_per_mbox
            ]
            percent = [x - x.mean() for x in percent]
            polymer.vecs = [
                x * mbox
                for x in itertools.product(*[[y for y in x] for x in percent])
            ]

        polymers = [x for x in self.polymers]
        idxs = range(
            math.ceil(math.pow(sum(x.num_mbox for x in polymers), 1. / 3)))
        vectors = [x * mbox for x in itertools.product(idxs, idxs, idxs)]
        mol_id = 0
        while polymers:
            random.shuffle(vectors)
            vector = vectors.pop()
            polymer = random.choice(polymers)
            for idx, new_added in enumerate(
                    range(min([polymer.mol_num, polymer.mol_num_per_mbox]))):
                mol_id += 1
                polymer.mol_num -= 1
                mol = copy.copy(polymer.polym)
                self.mols[mol_id] = mol
                mol.SetIntProp('mol_num', mol_id)
                conformer = mol.GetConformer(0)
                for atom in mol.GetAtoms():
                    atom_id = atom.GetIdx()
                    xyz = list(conformer.GetAtomPosition(atom_id))
                    xyz += (vector + polymer.vecs[idx])
                    conformer.SetAtomPosition(atom_id, xyz)
            if polymer.mol_num == 0:
                polymers.remove(polymer)

    def write(self):
        lmw = oplsua.LammpsWriter(self.ff, self.jobname, mols=self.mols)
        lmw.writeLammpsData()
        lmw.writeLammpsIn()


class Polymer(object):
    ATOM_ID = oplsua.LammpsWriter.ATOM_ID
    TYPE_ID = oplsua.LammpsWriter.TYPE_ID
    BOND_ATM_ID = oplsua.LammpsWriter.BOND_ATM_ID
    RES_NUM = oplsua.LammpsWriter.RES_NUM
    NEIGHBOR_CHARGE = oplsua.LammpsWriter.NEIGHBOR_CHARGE
    MOL_NUM = 'mol_num'
    IMPLICIT_H = oplsua.LammpsWriter.IMPLICIT_H

    def __init__(self, cru, cru_num, ff=None):
        self.cru = cru
        self.cru_num = cru_num
        self.ff = ff
        self.polym = None
        self.polym_Hs = None
        self.mols = {}
        self.buffer = oplsua.LammpsWriter.BUFFER
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()
        self.cru_mol = None
        self.molecules = []

    def run(self):
        self.setCruMol()
        self.markHT()
        self.polymerize()
        self.assignAtomType()
        self.balanceCharge()
        self.embedMol()
        # self.setMols()
        # self.write()
        # log('Finished', timestamp=True)

    def setCruMol(self):

        cru_mol = Chem.MolFromSmiles(self.cru)
        for atom in cru_mol.GetAtoms():
            if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
                continue
            atom.SetIntProp(self.IMPLICIT_H, atom.GetNumImplicitHs())
            atom.SetNoImplicit(True)
        self.cru_mol = Chem.AddHs(cru_mol)

    def markHT(self):
        is_mono = False
        for atom in self.cru_mol.GetAtoms():
            atom.SetIntProp('mono_atom_idx', atom.GetIdx())
            if atom.GetSymbol() != symbols.WILD_CARD:
                continue
            is_mono = True
            atom.SetBoolProp('CAP', True)
            for neighbor in atom.GetNeighbors():
                neighbor.SetBoolProp('HT', True)
        self.cru_mol.SetBoolProp('is_mono', is_mono)

    def polymerize(self):

        if not self.cru_mol.GetBoolProp('is_mono'):
            # FIXME: We should should polymers and blends with regular molecules
            self.polym = self.cru_mol
            return

        mols = [copy.copy(self.cru_mol) for x in range(self.cru_num)]
        for mono_id, mol in enumerate(mols):
            for atom in mol.GetAtoms():
                atom.SetIntProp('mono_id', mono_id)
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
        marked_atom_ids = []
        res_num = 1
        for sml in self.ff.SMILES:
            if all(x.HasProp(self.TYPE_ID) for x in self.polym.GetAtoms()):
                log_debug(f"{res_num - 1} residues found.")
                return
            frag = Chem.MolFromSmiles(sml.sml)
            matches = self.polym.GetSubstructMatches(frag, maxMatches=1000000)
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
                    marked_atom_ids += succeed
        log_debug(f"{sorted(marked_atom_ids)}, {self.polym.GetNumAtoms()}")
        log_debug(f"{res_num - 1} residues found.")

    def markAtoms(self, match, sml, res_num):
        marked = []
        for atom_id, type_id in zip(match, sml.mp):
            if not type_id or atom_id is None:
                continue
            atom = self.polym.GetAtomWithIdx(atom_id)
            try:
                atom.GetIntProp(self.TYPE_ID)
            except KeyError:
                self.setAtomIds(atom, type_id, res_num)
                marked.append(atom_id)
                log_debug(
                    f"{atom.GetSymbol()}{atom.GetDegree()} {atom_id} {type_id}"
                )
            else:
                continue
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    type_id = sml.hs[type_id]
                    self.setAtomIds(neighbor, type_id, res_num)
                    marked.append(neighbor.GetIdx())
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

    def embedMol(self, trans=False):

        if self.polym.GetNumAtoms() <= 200 and not trans:
            AllChem.EmbedMolecule(self.polym, useRandomCoords=True)
            return

        cru_mol = self.getCruMol()
        xyzs, vector = self.getXYZAndVect(cru_mol)
        conformer = self.getConformer(xyzs, vector)
        self.polym.AddConformer(conformer)

    def getCruMol(self):
        cru_mol = copy.copy(self.cru_mol)
        for atom in cru_mol.GetAtoms():
            if atom.GetSymbol() != symbols.WILD_CARD:
                continue
            atom.SetAtomicNum(6)
        return cru_mol

    def getXYZAndVect(self, mol):
        bk_dihes = self.getBackbone(mol)
        AllChem.EmbedMolecule(mol)
        conformer = mol.GetConformer(0)
        for dihe in zip(bk_dihes[:-3], bk_dihes[1:-2], bk_dihes[2:-1],
                        bk_dihes[3:]):
            Chem.rdMolTransforms.SetDihedralDeg(conformer, *dihe, 180)

        bonded_atom_ids = [(
            x.GetBeginAtomIdx(),
            x.GetEndAtomIdx(),
        ) for x in mol.GetBonds()]
        bk_aids_set = set(bk_dihes)
        side_atom_ids = [
            x for x in bonded_atom_ids if len(bk_aids_set.intersection(x)) == 1
        ]
        side_dihes = []
        for batom_id, eatom_id in side_atom_ids:
            id1 = [
                x.GetIdx()
                for x in mol.GetAtomWithIdx(batom_id).GetNeighbors()
                if x.GetIdx() != eatom_id
            ][0]
            id4 = [
                x.GetIdx()
                for x in mol.GetAtomWithIdx(eatom_id).GetNeighbors()
                if x.GetIdx() != batom_id
            ][0]
            side_dihes.append([id1, batom_id, eatom_id, id4])
        for dihe_atom_ids in side_dihes:
            Chem.rdMolTransforms.SetDihedralDeg(conformer, *dihe_atom_ids, 90)

        cap_ht = [(
            x.GetIdx(),
            [y.GetIdx() for y in x.GetNeighbors()][0],
        ) for x in mol.GetAtoms() if x.HasProp('CAP')]
        middle_points = np.array([
            (conformer.GetAtomPosition(x) + conformer.GetAtomPosition(y)) / 2
            for x, y in cap_ht
        ])
        vector = middle_points[1, :] - middle_points[0, :]
        xyzs = {
            x.GetIntProp('mono_atom_idx'):
            np.array(conformer.GetAtomPosition(x.GetIdx()))
            for x in mol.GetAtoms() if x.HasProp('mono_atom_idx')
        }
        return xyzs, vector

    def getBackbone(self, cru_mol):
        cap_idxs = [x.GetIdx() for x in cru_mol.GetAtoms() if x.HasProp('CAP')]
        graph = nx.Graph()
        edges = [(
            x.GetBeginAtom().GetIdx(),
            x.GetEndAtom().GetIdx(),
        ) for x in cru_mol.GetBonds()]
        graph.add_edges_from(edges)
        bk_dihes = nx.shortest_path(graph, *cap_idxs)
        return bk_dihes

    def getConformer(self, xyzs, vector):
        mid_aid = collections.defaultdict(list)
        for atom in self.polym.GetAtoms():
            mid_aid[atom.GetIntProp('mono_id')].append(atom.GetIdx())

        id_coords = {}
        for mono_id, atom_id in mid_aid.items():
            aid_oid = {
                x: self.polym.GetAtomWithIdx(x).GetIntProp('mono_atom_idx')
                for x in atom_id
            }
            vect = mono_id * vector
            aid_xyz = {x: xyzs[y] + vect for x, y in aid_oid.items()}
            id_coords.update(aid_xyz)

        conformer = Chem.rdchem.Conformer(self.polym.GetNumAtoms())
        for id, xyz in id_coords.items():
            conformer.SetAtomPosition(id, xyz)
        return conformer

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
    cell = AmorphousCell(options, jobname)
    cell.run()


if __name__ == "__main__":
    main(sys.argv[1:])
