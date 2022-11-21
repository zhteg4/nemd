import math
import copy
import lammps
import sys
import argparse
import random
import networkx as nx
import logutils
import functools
import os
import sys
import pandas as pd

import opls
import units
import traj
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
from scipy.spatial.transform import Rotation
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
        self.setBoxes()
        self.setPolymVectors()
        self.placeMols()

    def setBoxes(self):
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

        self.boxes = np.array(boxes).reshape(-1, 3)

    def setPolymVectors(self):
        mbox = self.boxes.max(axis=0)
        for polymer, box, mol_num in zip(self.polymers, self.boxes,
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

    def placeMols(self):
        polymers = [x for x in self.polymers]
        idxs = range(
            math.ceil(math.pow(sum(x.num_mbox for x in polymers), 1. / 3)))
        mbox = self.boxes.max(axis=0)
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
        lmw.writeLammpsData(adjust_bond_legnth=False)
        lmw.writeLammpsIn()


class Polymer(object):
    ATOM_ID = oplsua.LammpsWriter.ATOM_ID
    TYPE_ID = oplsua.LammpsWriter.TYPE_ID
    BOND_ATM_ID = oplsua.LammpsWriter.BOND_ATM_ID
    RES_NUM = oplsua.LammpsWriter.RES_NUM
    NEIGHBOR_CHARGE = oplsua.LammpsWriter.NEIGHBOR_CHARGE
    MOL_NUM = 'mol_num'
    IMPLICIT_H = oplsua.LammpsWriter.IMPLICIT_H

    def __init__(self, cru, cru_num, ff=None, srelaxation=True):
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

        chiralty_info = Chem.FindMolChiralCenters(cru_mol,
                                                  includeUnassigned=True)
        for chiralty in chiralty_info:
            cru_mol.GetAtomWithIdx(chiralty[0]).SetProp('_CIPCode', 'R')

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

    def embedMol(self, trans=True):

        if self.polym.GetNumAtoms() <= 200 and not trans:
            AllChem.EmbedMolecule(self.polym, useRandomCoords=True)
            return

        trans_conf = TransConformer(self.polym, self.cru_mol)
        trans_conf.run()

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


class TransConformer(object):

    def __init__(self, polym, original_cru_mol, ff=None, relaxation=True):
        self.polym = polym
        self.original_cru_mol = original_cru_mol
        self.ff = ff
        self.relaxation = relaxation
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()
        self.relax_dir = '_relax'
        self.data_file = 'data.polym'

    def run(self):
        self.setCruMol()
        self.cruConformer()
        self.setBackbone()
        self.setTransAndRotate()
        self.rotateSideGroups()
        self.setXYZAndVect()
        self.setConformer()
        self.adjustConformer()
        self.lammpsRelaxation()
        self.foldPolym()

    def setCruMol(self):
        cru_mol = copy.copy(self.original_cru_mol)
        atoms = [
            x for x in cru_mol.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
        ]
        neighbors = [x.GetNeighbors()[0] for x in atoms[::-1]]
        for atom, catom in zip(atoms, neighbors):
            atom.SetAtomicNum(catom.GetAtomicNum())
            atom.SetBoolProp('CAP', True)
        self.cru_mol = cru_mol

    def cruConformer(self):
        AllChem.EmbedMolecule(self.cru_mol)
        self.cru_conformer = self.cru_mol.GetConformer(0)

    def setBackbone(self, cru_mol=None):
        if cru_mol is None:
            cru_mol = self.cru_mol
        cap_idxs = [x.GetIdx() for x in cru_mol.GetAtoms() if x.HasProp('CAP')]
        if len(cap_idxs) != 2:
            raise ValueError(f'{len(cap_idxs)} capping atoms are found.')
        graph = nx.Graph()
        edges = [(
            x.GetBeginAtom().GetIdx(),
            x.GetEndAtom().GetIdx(),
        ) for x in cru_mol.GetBonds()]
        graph.add_edges_from(edges)
        bk_dihes = nx.shortest_path(graph, *cap_idxs)
        self.cru_bk_atom_ids = bk_dihes

    def setTransAndRotate(self):

        for dihe in zip(self.cru_bk_atom_ids[:-3], self.cru_bk_atom_ids[1:-2],
                        self.cru_bk_atom_ids[2:-1], self.cru_bk_atom_ids[3:]):
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conformer, *dihe, 180)

        bh_xyzs = np.array(
            [self.cru_conformer.GetAtomPosition(x) for x in dihe])
        centroid = bh_xyzs.mean(axis=0)
        for atom_id in range(self.cru_conformer.GetNumAtoms()):
            atom_pos = self.cru_conformer.GetAtomPosition(atom_id) - centroid
            self.cru_conformer.SetAtomPosition(atom_id, atom_pos)

        bvectors = (bh_xyzs[1:, :] - bh_xyzs[:-1, :])
        nc_vector = bvectors[::2].mean(axis=0)
        nc_vector /= np.linalg.norm(nc_vector)
        nm_mvector = bvectors[1::2].mean(axis=0)
        nm_mvector /= np.linalg.norm(nm_mvector)
        avect_norm = nc_vector + nm_mvector
        avect_norm /= np.linalg.norm(avect_norm)
        bvect_norm = nc_vector - nm_mvector
        bvect_norm /= np.linalg.norm(bvect_norm)
        cvect_norm = np.cross(avect_norm, bvect_norm)
        cvect_norm /= np.linalg.norm(cvect_norm)
        abc_norms = np.concatenate([
            avect_norm.reshape(1, -1),
            bvect_norm.reshape(1, -1),
            cvect_norm.reshape(1, -1)
        ],
                                   axis=0)
        abc_targeted = np.eye(3)
        rotation, rmsd = Rotation.align_vectors(abc_targeted, abc_norms)
        for atom_id in range(self.cru_conformer.GetNumAtoms()):
            atom_pos = self.cru_conformer.GetAtomPosition(atom_id)
            self.cru_conformer.SetAtomPosition(atom_id,
                                               rotation.apply(atom_pos))

    def rotateSideGroups(self):

        bonded_atom_ids = [(
            x.GetBeginAtomIdx(),
            x.GetEndAtomIdx(),
        ) for x in self.cru_mol.GetBonds()]
        bk_aids_set = set(self.cru_bk_atom_ids)
        side_atom_ids = [
            x for x in bonded_atom_ids if len(bk_aids_set.intersection(x)) == 1
        ]
        side_dihes = []
        for batom_id, eatom_id in side_atom_ids:
            id1 = [
                x.GetIdx()
                for x in self.cru_mol.GetAtomWithIdx(batom_id).GetNeighbors()
                if x.GetIdx() != eatom_id
            ][0]
            id4 = [
                x.GetIdx()
                for x in self.cru_mol.GetAtomWithIdx(eatom_id).GetNeighbors()
                if x.GetIdx() != batom_id
            ][0]
            side_dihes.append([id1, batom_id, eatom_id, id4])
        for dihe_atom_ids in side_dihes:
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conformer,
                                                *dihe_atom_ids, 90)

    def setXYZAndVect(self):

        cap_ht = [(
            x.GetIdx(),
            [y.GetIdx() for y in x.GetNeighbors()][0],
        ) for x in self.cru_mol.GetAtoms() if x.HasProp('CAP')]
        middle_points = np.array([(self.cru_conformer.GetAtomPosition(x) +
                                   self.cru_conformer.GetAtomPosition(y)) / 2
                                  for x, y in cap_ht])
        self.vector = middle_points[1, :] - middle_points[0, :]
        self.xyzs = {
            x.GetIntProp('mono_atom_idx'):
            np.array(self.cru_conformer.GetAtomPosition(x.GetIdx()))
            for x in self.cru_mol.GetAtoms() if x.HasProp('mono_atom_idx')
        }

    def setConformer(self):
        mid_aid = collections.defaultdict(list)
        for atom in self.polym.GetAtoms():
            mid_aid[atom.GetIntProp('mono_id')].append(atom.GetIdx())

        id_coords = {}
        for mono_id, atom_id in mid_aid.items():
            aid_oid = {
                x: self.polym.GetAtomWithIdx(x).GetIntProp('mono_atom_idx')
                for x in atom_id
            }
            vect = mono_id * self.vector
            aid_xyz = {x: self.xyzs[y] + vect for x, y in aid_oid.items()}
            id_coords.update(aid_xyz)

        conformer = Chem.rdchem.Conformer(self.polym.GetNumAtoms())
        for id, xyz in id_coords.items():
            conformer.SetAtomPosition(id, xyz)
        self.polym.AddConformer(conformer)
        Chem.GetSymmSSSR(self.polym)

    def adjustConformer(self):
        self.lmw = oplsua.LammpsWriter(self.ff,
                                       'myjobname',
                                       mols={1: self.polym})
        if not self.relaxation:
            self.lmw.adjustConformer()
            return
        with fileutils.chdir(self.relax_dir):
            self.lmw.writeLammpsData()
            self.lmw.writeLammpsIn()

    def lammpsRelaxation(self, min_cycle=10, max_cycle=100, threshold=.99):

        with fileutils.chdir(self.relax_dir):
            lmp = lammps.lammps(cmdargs=['-screen', 'none'])
            lmp.file(self.lmw.lammps_in)
            lmp.command(f'write_data {self.data_file}')
            # import pdb;pdb.set_trace()
            # lmp.command('compute 1 all gyration')
            # data = []
            # for iclycle in range(max_cycle):
            #     lmp.command('run 1000')
            #     data.append(lmp.extract_compute('1', 0, 0))
            #     if iclycle >= min_cycle:
            #         percentage = np.mean(data[-5:]) / np.mean(data[-10:])
            #         if percentage >= threshold:
            #             break

    def foldPolym(self):
        data_file = os.path.join(self.relax_dir, self.data_file)
        self.data_reader = oplsua.DataFileReader(data_file)
        self.data_reader.run()
        self.data_reader.setClashParams(include14=True, scale=0.6)

        conformer = self.polym.GetConformer()
        for atom in self.data_reader.atoms.values():
            conformer.SetAtomPosition(atom.id - 1, np.array(atom.xyz))

        self.setBackboneDihedrals()
        self.setFragAtoms()

        box = np.array(
            [y for x in self.data_reader.box_dsp.values() for y in x])
        frm = pd.DataFrame(conformer.GetPositions(),
                           index=range(1,
                                       conformer.GetNumAtoms() + 1),
                           columns=['xu', 'yu', 'zu'])
        frm.attrs['box'] = box
        random.seed(2022)

        for dihe, vals in self.bk_dihe_atom_ids.items():
            while (len(vals)):
                random.shuffle(vals)
                val = vals.pop()
                Chem.rdMolTransforms.SetDihedralDeg(conformer, *dihe, val)
                frm.loc[:] = conformer.GetPositions()
                dcell = traj.DistanceCell(frm=frm, cut=10, resolution=2.)
                dcell.setUp()
                clashes = []
                # frag_atom_ids = self.frag_atom_ids[dihe]
                # frag_rows = [frm.iloc[x] for x in frag_atom_ids]
                # import pdb;pdb.set_trace()
                for _, row in frm.iterrows():
                    clashes += dcell.getClashes(
                        row,
                        # included= [x for x in frag_atom_ids]
                        radii=self.data_reader.radii,
                        excluded=self.data_reader.excluded)
                if not clashes:
                    break
        #         if clashes and not vals:
        #             import pdb;pdb.set_trace()
        # import pdb;
        # pdb.set_trace()

    def setBackboneDihedrals(self):
        cru_bk_atom_ids = set(self.cru_bk_atom_ids)
        self.backbone_atoms = [
            x for x in self.polym.GetAtoms()
            if int(x.GetProp('mono_atom_idx')) in cru_bk_atom_ids
        ]
        assert all([
            x.GetIdx() in [z.GetIdx() for z in y.GetNeighbors()]
            for x, y in zip(self.backbone_atoms[1:], self.backbone_atoms[:-1])
        ])
        assert all([
            x.GetIdx() in [z.GetIdx() for z in y.GetNeighbors()]
            for x, y in zip(self.backbone_atoms[:-1], self.backbone_atoms[1:])
        ])
        self.bk_dihe_atom_ids = zip(self.backbone_atoms[:-3],
                                    self.backbone_atoms[1:-2],
                                    self.backbone_atoms[2:-1],
                                    self.backbone_atoms[3:])
        self.bk_dihe_atom_ids = {
            tuple(y.GetIdx()
                  for y in x): list(np.linspace(0, 360, 36, endpoint=False))
            for x in self.bk_dihe_atom_ids
        }

    def setFragAtoms(self):
        self.all_frag_atom_ids = {}
        conformer = self.polym.GetConformer()
        for dihe_atom_ids in self.bk_dihe_atom_ids.keys():
            orgin_xyz = conformer.GetPositions()
            orgin_val = Chem.rdMolTransforms.GetDihedralDeg(
                conformer, *dihe_atom_ids)
            Chem.rdMolTransforms.SetDihedralDeg(conformer, *dihe_atom_ids,
                                                orgin_val + 5)
            xyz = conformer.GetPositions()
            changed = np.isclose(orgin_xyz, xyz)
            self.all_frag_atom_ids[dihe_atom_ids] = [
                i for i, x in enumerate(changed) if not all(x)
            ]
            Chem.rdMolTransforms.SetDihedralDeg(conformer, *dihe_atom_ids,
                                                orgin_val)

        moved = set()
        self.frag_atom_ids = {}
        for dihe in reversed(self.bk_dihe_atom_ids.keys()):
            all_frag_atom_ids = self.all_frag_atom_ids[dihe]
            self.frag_atom_ids[dihe] = set(all_frag_atom_ids).difference(moved)
            moved = moved.union(self.frag_atom_ids[dihe])
        self.frag_atom_ids = {
            x: list(self.frag_atom_ids[x])
            for x in self.bk_dihe_atom_ids.keys()
        }

        self.existing_atom_ids = set(x for x in range(conformer.GetNumAtoms()))
        self.existing_atom_ids = list(self.existing_atom_ids.difference(moved))

    def write(self):

        with Chem.SDWriter('polym.sdf') as polym_fh:
            polym_fh.SetProps(["mono_atom_idxs"])
            mono_atom_idxs = [
                x.GetIntProp('mono_atom_idx') for x in self.polym.GetAtoms()
            ]
            self.polym.SetProp('mono_atom_idxs',
                               ' '.join(map(str, mono_atom_idxs)))
            polym_fh.write(self.polym)

        with Chem.SDWriter('original_cru_mol.sdf') as cru_fh:
            cru_fh.write(self.original_cru_mol)

    @staticmethod
    def read(filename):
        suppl = Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
        mol = next(suppl)
        # mol = Chem.AddHs(mol)
        Chem.GetSymmSSSR(mol)
        try:
            mono_atom_idxs = mol.GetProp('mono_atom_idxs').split()
        except KeyError:
            return mol
        for atom, mono_atom_idx in zip(mol.GetAtoms(), mono_atom_idxs):
            atom.SetProp('mono_atom_idx', mono_atom_idx)
        return mol


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
