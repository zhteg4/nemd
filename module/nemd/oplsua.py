# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles opls-ua related typing, parameterization, assignment,
datafile, and in-script.
"""
import chemparse
import itertools
import collections
import numpy as np
from rdkit import Chem
from collections import namedtuple

from nemd import symbols
from nemd import logutils
from nemd import fileutils
from nemd import constants as nconstants

ATOM_TYPE = namedtuple('ATOM_TYPE', [
    'id', 'formula', 'symbol', 'description', 'atomic_number', 'mass', 'conn'
])
VDW = namedtuple('VDW', ['id', 'dist', 'ene'])
BOND = namedtuple('BOND', ['id', 'id1', 'id2', 'dist', 'ene', 'has_h'])
ANGLE = namedtuple('ANGLE', ['id', 'id1', 'id2', 'id3', 'ene', 'deg', 'has_h'])
UREY_BRADLEY = namedtuple('UREY_BRADLEY', ['id1', 'id2', 'id3', 'ene', 'dist'])
IMPROPER = namedtuple(
    'IMPROPER', ['id', 'id1', 'id2', 'id3', 'id4', 'ene', 'deg', 'n_parm'])
ENE_ANG_N = namedtuple('ENE_ANG_N', ['ene', 'deg', 'n_parm'])
DIHEDRAL = namedtuple('DIHEDRAL',
                      ['id', 'id1', 'id2', 'id3', 'id4', 'constants'])

UA = namedtuple('UA', ['sml', 'mp', 'hs', 'dsc'])

RES_NUM = 'res_num'
TYPE_ID = 'type_id'
BOND_AID = 'bond_aid'
ANGLE_ATM_ID = 'angle_atm_id'
DIHE_ATM_ID = 'dihe_atm_id'
IMPLICIT_H = symbols.IMPLICIT_H
LARGE_NUM = nconstants.LARGE_NUM

# https://docs.lammps.org/Howto_tip3p.html
SPC = symbols.SPC
SPCE = symbols.SPCE
TIP3P = symbols.TIP3P
WATER_TIP3P = f'Water ({TIP3P})'
WATER_SPC = f'Water ({SPC})'
WATER_SPCE = f'Water ({SPCE})'
UA_WATER_TIP3P = UA(sml='O', mp=(77, ), hs={77: 78}, dsc='Water (TIP3P)')
UA_WATER_SPC = UA(sml='O', mp=(79, ), hs={79: 80}, dsc=WATER_SPC)
UA_WATER_SPCE = UA(sml='O', mp=(214, ), hs={214: 215}, dsc=WATER_SPCE)
WMODELS = {TIP3P: UA_WATER_TIP3P, SPC: UA_WATER_SPC, SPCE: UA_WATER_SPCE}
# yapf: disable
SMILES_TEMPLATE = [
    # Single Atom Particle
    UA(sml='[Li+]', mp=(197,), hs=None, dsc='Li+ Lithium Ion'),
    UA(sml='[Na+]', mp=(198,), hs=None, dsc='Na+ Sodium Ion'),
    UA(sml='[K+]', mp=(199,), hs=None, dsc='K+ Potassium Ion'),
    UA(sml='[Rb+]', mp=(200,), hs=None, dsc='Rb+ Rubidium Ion'),
    UA(sml='[Cs+]', mp=(201,), hs=None, dsc='Cs+ Cesium Ion'),
    UA(sml='[Mg+2]', mp=(202,), hs=None, dsc='Mg+2 Magnesium Ion'),
    UA(sml='[Ca+2]', mp=(203,), hs=None, dsc='Ca+2 Calcium Ion'),
    UA(sml='[Sr+2]', mp=(204,), hs=None, dsc='Sr+2 Strontium Ion'),
    UA(sml='[Ba+2]', mp=(205,), hs=None, dsc='Ba+2 Barium Ion'),
    UA(sml='[F-]', mp=(206,), hs=None, dsc='F- Fluoride Ion'),
    UA(sml='[Cl-]', mp=(207,), hs=None, dsc='Cl- Chloride Ion'),
    UA(sml='[Br-]', mp=(208,), hs=None, dsc='Br- Bromide Ion'),
    UA(sml='[He]', mp=(209,), hs=None, dsc='Helium Atom'),
    UA(sml='[Ne]', mp=(210,), hs=None, dsc='Neon Atom'),
    UA(sml='[Ar]', mp=(211,), hs=None, dsc='Argon Atom'),
    UA(sml='[Kr]', mp=(212,), hs=None, dsc='Krypton Atom'),
    UA(sml='[Xe]', mp=(213,), hs=None, dsc='Xenon Atom'),
    # Alkane
    UA(sml='C', mp=(81,), hs=None, dsc='CH4 Methane'),
    UA(sml='CC', mp=(82, 82,), hs=None, dsc='Ethane'),
    UA(sml='CCC', mp=(83, 86, 83,), hs=None, dsc='Propane'),
    UA(sml='CCCC', mp=(83, 86, 86, 83,), hs=None, dsc='n-Butane'),
    UA(sml='CC(C)C', mp=(84, 88, 84, 84,), hs=None, dsc='Isobutane'),
    UA(sml='CC(C)(C)C', mp=(85, 90, 85, 85, 85,), hs=None, dsc='Neopentane'),
    # Alkene
    UA(sml='CC=CC', mp=(84, 89, 89, 84,), hs=None, dsc='2-Butene'),
    # Aldehydes (with formyl group)
    # Ketone
    UA(sml='CC(=O)C', mp=(129, 127, 128, 129,), hs=None, dsc='Acetone'),
    UA(sml='CCC(=O)CC', mp=(7, 130, 127, 128, 130, 7,), hs=None,
       dsc='Diethyl Ketone'),
    # t-Butyl Ketone CC(C)CC(=O)C(C)(C)C described by Neopentane, Acetone, and Diethyl Ketone
    # Alcohol
    UA_WATER_TIP3P,
    UA(sml='CO', mp=(106, 104,), hs={104: 105}, dsc='Methanol'),
    UA(sml='CCO', mp=(83, 107, 104,), hs={104: 105}, dsc='Ethanol'),
    UA(sml='CC(C)O', mp=(84, 108, 84, 104,), hs={104: 105}, dsc='Isopropanol'),
    # Carboxylic Acids
    # "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
    UA(sml='O=CO', mp=(134, 133, 135), hs={135: 136}, dsc='Carboxylic Acid'),
    # "Methyl", "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
    UA(sml='CC(=O)O', mp=(137, 133, 134, 135), hs={135: 136},
       dsc='Ethanoic acid'),
    # Large Molecules
    UA(sml='CN(C)C=O', mp=(156, 148, 156, 153, 151,), hs=None,
       dsc='N,N-Dimethylformamide')
]

ATOM_TOTAL = {i: i for i in range(1, 216)}
BOND_ATOM = ATOM_TOTAL.copy()
# "O Peptide Amide" "COH (zeta) Tyr" "OH Tyr"  "H(O) Ser/Thr/Tyr"
BOND_ATOM.update({134: 2, 133: 26, 135: 23, 136: 24, 153: 72, 148: 3,
                  108: 107, 127: 1, 128: 2, 129: 7, 130: 9, 85: 9, 90: 64})
ANGLE_ATOM = ATOM_TOTAL.copy()
ANGLE_ATOM.update({134: 2, 133: 17, 135: 76, 136: 24, 148: 3, 153: 72,
                   108: 107, 127: 1, 129: 7, 130: 9})
DIHE_ATOM = ATOM_TOTAL.copy()
DIHE_ATOM.update({134: 11, 133: 26, 135: 76, 136: 24, 148: 3, 153: 72,
                  108: 107, 127: 1, 130: 9, 86: 9, 88: 9, 90: 9})
# C-OH (Tyr) is used as HO-C=O, which needs CH2-COOH map as alpha-COOH bond
BOND_ATOMS = {(26, 86): [16, 17], (26, 88): [16, 17], (86, 107): [86, 86]}
ANGLE_ATOMS = {(84, 107, 84): (86, 88, 86), (84, 107, 86): (86, 88, 83),
               (86, 107, 86): (86, 88, 83)}
DIHE_ATOMS = {(26, 86,): (1, 6,), (26, 88,): (1, 6,), (88, 107,): (6, 22,),
              (86, 107,): (6, 25,), (9, 26): (1, 9), (9, 107): (9, 9)}
# yapf: enable

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


def get_parser(wmodel=symbols.TIP3P):
    """
    Read and parser opls force field file.

    :param wmodel str: the model type for water
    :return 'OplsParser': the parser with force field information
    """
    parser = Parser(wmodel=wmodel)
    parser.read()
    return parser


class Typer:
    """
    Type the atoms and map SMILES fragments.
    """

    def __init__(self, mol, wmodel=TIP3P):
        """
        :param mol 'rdkit.Chem.rdchem.Mol': molecule to assign FF types
        :param wmodel str: the model type for water
        """
        self.mol = mol
        self.SMILES = list(reversed(SMILES_TEMPLATE))
        if wmodel == TIP3P:
            return
        id = next(x for x, y in enumerate(self.SMILES) if y.dsc == WATER_TIP3P)
        self.SMILES[id] = WMODELS[wmodel]

    def run(self):
        """
        Assign atom types for force field assignment.
        """
        self.doTyping()
        self.reassignResnum()

    def doTyping(self):
        """
        Match the substructure with SMILES and assign atom type.
        """
        marked_smiles = {}
        marked_atom_ids = []
        res_num = 1
        for sml in self.SMILES:
            frag = Chem.MolFromSmiles(sml.sml)
            matches = self.mol.GetSubstructMatches(frag, maxMatches=LARGE_NUM)
            matches = [self.filterMatch(x, frag) for x in matches]
            res_num, matom_ids = self.markMatches(matches, sml, res_num)
            if not matom_ids:
                continue
            cnt = collections.Counter([len(x) for x in matom_ids])
            cnt_exp = str(len(matom_ids)) + ' matches ' + ','.join(
                [f'{x}*{y}' for x, y in cnt.items()])
            marked_smiles[sml.sml] = cnt_exp
            marked_atom_ids += [y for x in matom_ids for y in x]
            if all(x.HasProp(TYPE_ID) for x in self.mol.GetAtoms()):
                break
        log_debug(
            f"{len(marked_atom_ids)} out of {self.mol.GetNumAtoms()} atoms marked"
        )
        log_debug(f"{res_num - 1} residues found.")
        [log_debug(f'{x}: {y}') for x, y in marked_smiles.items()]

    def reassignResnum(self):
        """
        Reassign residue number based on the fragments (SMILES match results).
        """
        res_atom = collections.defaultdict(list)
        for atom in self.mol.GetAtoms():
            try:
                res_num = atom.GetIntProp(RES_NUM)
            except KeyError:
                raise KeyError(
                    f'Typing missed for {atom.GetSymbol()} atom {atom.GetIdx()}'
                )
            res_atom[res_num].append(atom.GetIdx())
        cbonds = [
            x for x in self.mol.GetBonds() if x.GetBeginAtom().GetIntProp(
                RES_NUM) != x.GetEndAtom().GetIntProp(RES_NUM)
        ]
        emol = Chem.EditableMol(Chem.Mol(self.mol))
        [
            emol.RemoveBond(x.GetBeginAtom().GetIdx(),
                            x.GetEndAtom().GetIdx()) for x in cbonds
        ]
        frags = Chem.GetMolFrags(emol.GetMol())
        [
            self.mol.GetAtomWithIdx(y).SetIntProp(RES_NUM, i)
            for i, x in enumerate(frags, 1) for y in x
        ]
        log_debug(f"{len(frags)} residues reassigned.")

    def filterMatch(self, match, frag):
        """
        Filter substructure matches based on connectivity. The connecting atoms
        usually have different connectivities. For example, first C in 'CC(=O)O'
        fragment terminates while the second 'C' in 'CCC(=O)O' molecule is
        connected to two carbons. Mark the first C in 'CC(=O)O' fragment as None
        so that molecule won't type this terminating atom.

        :param match tuples: atom ids of one match
        :param frag: the fragment of one force field templated smiles
        :return: tuples: atom ids of one match with correct connectivity
        """
        frag_cnnt = [
            x.GetNumImplicitHs() + x.GetDegree()
            if x.GetSymbol() != symbols.CARBON else x.GetDegree()
            for x in frag.GetAtoms()
        ]
        polm_cnnt = [self.mol.GetAtomWithIdx(x).GetDegree() for x in match]
        match = [
            x if y == z else None
            for x, y, z in zip(match, frag_cnnt, polm_cnnt)
        ]
        return match

    def markMatches(self, matches, sml, res_num):
        """
        Mark the matched atoms.

        :param matches list of tuple: each tuple has one pattern match
        :param sml namedtuple: 'UA' namedtuple for smiles
        :param res_num int: the residue number
        :return int, list: incremented residue number, list of marked atom list
        """
        marked_atom_ids = []
        for match in matches:
            log_debug(f"assignAtomType {sml.sml}, {match}")
            marked = self.markAtoms(match, sml, res_num)
            if marked:
                res_num += 1
                marked_atom_ids.append(marked)
        return res_num, marked_atom_ids

    def markAtoms(self, match, sml, res_num):
        """
        Marker atoms with type id, res_num, and bonded_atom id for vdw/charge
            table lookup, charge balance, and bond searching.

        :param match tuple: atom ids of one match
        :param sml namedtuple: 'UA' namedtuple for smiles
        :param res_num int: the residue number
        :return list: list of marked atom ids
        """
        marked = []
        for atom_id, type_id in zip(match, sml.mp):
            if not type_id or atom_id is None:
                continue
            atom = self.mol.GetAtomWithIdx(atom_id)
            try:
                atom.GetIntProp(TYPE_ID)
            except KeyError:
                self.markAtom(atom, type_id, res_num)
                marked.append(atom_id)
                log_debug(
                    f"{atom.GetSymbol()}{atom.GetDegree()} {atom_id} {type_id}"
                )
            else:
                continue
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol() != symbols.HYDROGEN:
                    continue
                htype_id = sml.hs[type_id]
                self.markAtom(nbr, htype_id, res_num)
                marked.append(nbr.GetIdx())
                msg = f"{nbr.GetSymbol()}{nbr.GetDegree()} {nbr.GetIdx()} {htype_id}"
                log_debug(msg)
        return marked

    def markAtom(self, atom, type_id, res_num):
        """
        Set atom id, res_num, and bonded_atom id.

        :param atom 'rdkit.Chem.rdchem.Atom': the atom to mark
        :param type_id int: atom type id
        :param res_num int: residue number
        """

        # TYPE_ID defines vdw and charge
        atom.SetIntProp(TYPE_ID, type_id)
        atom.SetIntProp(RES_NUM, res_num)
        # BOND_AID defines bonding parameters
        atom.SetIntProp(BOND_AID, BOND_ATOM[type_id])
        atom.SetIntProp(ANGLE_ATM_ID, ANGLE_ATOM[type_id])
        atom.SetIntProp(DIHE_ATM_ID, DIHE_ATOM[type_id])


class Parser:
    """
    Parse force field file and map atomic details.
    """

    FILE_PATH = fileutils.get_ff(name=fileutils.OPLSUA)

    DEFINITION_MK = 'Force Field Definition'
    LITERATURE_MK = 'Literature References'
    ATOM_MK = 'Atom Type Definitions'
    VAN_MK = 'Van der Waals Parameters'
    BOND_MK = 'Bond Stretching Parameters'
    ANGLE_MK = 'Angle Bending Parameters'
    UREY_MK = 'Urey-Bradley Parameters'
    IMPROPER_MK = 'Improper Torsional Parameters'
    TORSIONAL_MK = 'Torsional Parameters'
    ATOMIC_MK = 'Atomic Partial Charge Parameters'
    BIOPOLYMER_MK = 'Biopolymer Atom Type Conversions'

    MARKERS = [
        DEFINITION_MK, LITERATURE_MK, ATOM_MK, VAN_MK, BOND_MK, ANGLE_MK,
        UREY_MK, IMPROPER_MK, TORSIONAL_MK, ATOMIC_MK, BIOPOLYMER_MK
    ]

    def __init__(self, wmodel=symbols.TIP3P):
        """
        :param filepath str: the path to the force field file.
        :param wmodel str: the model type for water
        """
        self.wmodel = wmodel
        self.raw_content = {}
        self.atoms = {}
        self.vdws = {}
        self.bonds = {}
        self.angles = {}
        self.urey_bradleys = {}
        self.impropers = {}
        self.dihedrals = {}
        self.charges = {}
        self.symbol_impropers = None
        self.bnd_map = None
        self.ang_map = None

    def type(self, mol):
        """
        Type the molecule based on the force field typer.

        :param mol 'rdkit.Chem.rdchem.Mol': the molecule to type.
        """
        Typer(mol, wmodel=self.wmodel).run()

    def read(self):
        """
        Main method to read and parse the force field file.
        """
        self.setRawContent()
        self.setAtomType()
        self.setVdW()
        self.setBond()
        self.setAngle()
        self.setUreyBradley()
        self.setImproper()
        self.setDihedral()
        self.setCharge()

    def setRawContent(self):
        """
        Read and set raw content.
        """

        with open(self.FILE_PATH, 'r') as fp:
            lns = [x.strip(' \n') for x in fp.readlines()]
        mls = {m: i for i, l in enumerate(lns) for m in self.MARKERS if m in l}
        for bmarker, emarker in zip(self.MARKERS[:-1], self.MARKERS[1:]):
            content_lines = lns[mls[bmarker]:mls[emarker]]
            self.raw_content[bmarker] = [
                x for x in content_lines
                if x and not x.startswith(symbols.POUND)
            ]

    def setAtomType(self):
        """
        Set atom types based on the 'Atom Type Definitions' block.
        """
        for line in self.raw_content[self.ATOM_MK]:
            # 'atom       1    C     "C Peptide Amide"         6    12.011    3'
            bcomment, comment, acomment = line.split(symbols.DOUBLE_QUOTATION)
            _, id, formula = bcomment.split()
            atomic_number, mass, cnnct = acomment.split()  # CH3, CH, C, H
            prsd = chemparse.parse_formula(formula)
            h_count = int(prsd.pop(symbols.HYDROGEN, 0))
            symbol = [x for x in prsd.keys()][0] if prsd else symbols.HYDROGEN
            self.atoms[int(id)] = ATOM_TYPE(id=int(id),
                                            formula=formula,
                                            symbol=symbol,
                                            description=comment,
                                            atomic_number=int(atomic_number),
                                            mass=float(mass),
                                            conn=int(cnnct) + h_count)

    def setVdW(self):
        """
        Set vdw parameters based on 'Van der Waals Parameters' block.
        """
        for line in self.raw_content[self.VAN_MK]:
            # 'vdw         213               2.5560     0.4330'
            _, id, dist, ene = line.split()
            self.vdws[int(id)] = VDW(id=int(id),
                                     dist=float(dist),
                                     ene=float(ene))

    def setCharge(self):
        """
        Set charges based on 'Atomic Partial Charge Parameters' block.
        """
        for line in self.raw_content[self.ATOMIC_MK]:
            # 'charge      213               0.0000'
            _, type_id, charge = line.split()
            self.charges[int(type_id)] = float(charge)

    def setBond(self):
        """
        Set bond parameters based on 'Bond Stretching Parameters' block.
        """
        shape = len(self.atoms) + 1
        self.bnd_map = np.zeros((shape, shape), dtype=np.int16)
        for id, line in enumerate(self.raw_content[self.BOND_MK], 1):
            # 'bond        104  107          386.00     1.4250'
            _, id1, id2, ene, dist = line.split()
            atoms = [self.atoms[int(x)] for x in [id1, id2]]
            has_h = any(x.symbol == symbols.HYDROGEN for x in atoms)
            self.bonds[id] = BOND(id=id,
                                  id1=int(id1),
                                  id2=int(id2),
                                  ene=float(ene),
                                  dist=float(dist),
                                  has_h=has_h)
            self.bnd_map[int(id1), int(id2)] = id

    def setAngle(self):
        """
        Set angle parameters based on 'Angle Bending Parameters' block.
        """
        shape = len(self.atoms) + 1
        self.ang_map = np.zeros((shape, shape, shape), dtype=np.int16)
        for id, line in enumerate(self.raw_content[self.ANGLE_MK], 1):
            # 'angle        83  107  104      80.00     109.50'
            _, id1, id2, id3, ene, angle = line.split()
            atoms = [self.atoms[int(x)] for x in [id1, id2, id3]]
            has_h = any(x.symbol == symbols.HYDROGEN for x in atoms)
            self.angles[id] = ANGLE(id=id,
                                    id1=int(id1),
                                    id2=int(id2),
                                    id3=int(id3),
                                    ene=float(ene),
                                    deg=float(angle),
                                    has_h=has_h)
            self.ang_map[int(id1), int(id2), int(id3)] = id

    def setUreyBradley(self):
        """
        Set parameters based on 'Urey-Bradley Parameters' block.

        NOTE: current this is not supported.
        """
        for id, line in enumerate(self.raw_content[self.UREY_MK], 1):
            # ureybrad     78   77   78      38.25     1.5139
            # ureybrad     80   79   80      39.90     1.6330
            _, id1, id2, id3, ene, dist = line.split()
            self.urey_bradleys[id] = UREY_BRADLEY(id1=int(id1),
                                                  id2=int(id2),
                                                  id3=int(id3),
                                                  ene=float(ene),
                                                  dist=float(dist))

    def setImproper(self):
        """
        Set improper parameters based on 'Improper Torsional Parameters' block.
        """
        for id, line in enumerate(self.raw_content[self.IMPROPER_MK], 1):
            # imptors       5    3    1    2           10.500  180.0  2
            _, id1, id2, id3, id4, ene, angle, n_parm = line.split()
            self.impropers[id] = IMPROPER(id=id,
                                          id1=int(id1),
                                          id2=int(id2),
                                          id3=int(id3),
                                          id4=int(id4),
                                          ene=float(ene),
                                          deg=float(angle),
                                          n_parm=int(n_parm))

    def setDihedral(self):
        """
        Set dihedral parameters based on 'Torsional Parameters' block.
        """
        shape = len(self.atoms) + 1
        self.dihe_map = np.zeros((shape, shape, shape, shape), dtype=np.int16)
        for id, line in enumerate(self.raw_content[self.TORSIONAL_MK], 1):
            # torsion       2    1    3    4            0.650    0.0  1      2.500  180.0  2
            line_splitted = line.split()
            ids, enes = line_splitted[1:5], line_splitted[5:]
            ids = list(map(int, ids))
            ene_ang_ns = tuple(
                ENE_ANG_N(ene=float(x), deg=float(y), n_parm=int(z))
                for x, y, z in zip(enes[::3], enes[1::3], enes[2::3]))
            self.dihedrals[id] = DIHEDRAL(id=id,
                                          id1=ids[0],
                                          id2=ids[1],
                                          id3=ids[2],
                                          id4=ids[3],
                                          constants=ene_ang_ns)
            self.dihe_map[ids[0], ids[1], ids[2], ids[3]] = id

    def getMatchedBonds(self, bond):
        """
        Get force field matched bonds. The searching and approximation follows:

        1) Forced mapping via BOND_ATOMS to connect force field fragments.
        2) Exact match for current atom types.
        3) Matching of one atom with the other's symbol and connectivity matched
        4) Matching of one atom with only the other's symbol matched

        :raise ValueError: If the above failed

        :param bonded_atoms: list of two bonded atoms sorted by BOND_AID
        :return list of 'oplsua.BOND': bond information
        """
        bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
        # BOND_AID defines bonding parameters marked during atom typing
        bonded_atoms = sorted(bonded_atoms,
                              key=lambda x: x.GetIntProp(BOND_AID))

        atypes = sorted([x.GetIntProp(BOND_AID) for x in bonded_atoms])
        try:
            atypes = BOND_ATOMS[tuple(atypes)]
        except KeyError:
            # C-OH (Tyr) is used as HO-C=O, needing CH2-COOH map as alpha-COOH bond
            pass
        try:
            # Exact match between two atom type ids
            return [self.bonds[self.bnd_map[atypes[0], atypes[1]]]]
        except KeyError:
            pass

        msg = f"No exact params for bond between atom type {atypes[0]} and {atypes[1]}."
        log_debug(msg)

        partial_matches = []
        for atype in atypes:
            matched = self.bnd_map[atype, :]
            partial_matches += list(matched[matched != 0])
            matched = self.bnd_map[:, atype]
            partial_matches += list(matched[matched != 0])

        bond_score, type_set = {}, set(atypes)
        for bond_id in partial_matches:
            bond = self.bonds[bond_id]
            matched = type_set.intersection([bond.id1, bond.id2])
            # Compare the unmatched and sore them
            try:
                atom_id = set([bond.id1, bond.id2]).difference(matched).pop()
            except KeyError:
                # bond.id1, bond.id2, matched are the same and thus the unmatch
                # bond.id1, bond.id2, and list(matched)[0] are the same
                atom_id = bond.id1
            atom = [
                x for x in bonded_atoms
                if x.GetIntProp(BOND_AID) not in [bond.id1, bond.id2]
            ][0]
            ssymbol = self.atoms[atom_id].symbol == atom.GetSymbol()
            scnnt = self.atoms[atom_id].conn == self.getAtomConnt(atom)
            bond_score[bond] = [ssymbol, scnnt]

        matches = [x for x, y_z in bond_score.items() if all(y_z)]
        if not matches:
            matches = [x for x, (y, z) in bond_score.items() if y]
        if not matches:
            err = f"No params for bond between atom type {atypes[0]} and {atypes[1]}."
            raise ValueError(err)
        self.debugPrintReplacement(bonded_atoms, matches)
        return matches

    @staticmethod
    def getAtomConnt(atom):
        """
        Get the atomic connectivity information.

        :param atom 'rdkit.Chem.rdchem.Atom': the connectivity of this atom
        :return int: the number of bonds connected to this atom including the
            implicit hydrogen.
        """

        implicit_h_num = atom.GetIntProp(IMPLICIT_H) if atom.HasProp(
            IMPLICIT_H) else atom.GetNumImplicitHs()
        return atom.GetDegree() + implicit_h_num

    def debugPrintReplacement(self, atoms, matches):
        """
        Print the debug information on matching approximation.

        :param atoms list of 'rdkit.Chem.rdchem.Atom': matched atoms
        :param matches list of namedtuple: forced information
        """

        smbl_cnnts = [f'{x.GetSymbol()}{self.getAtomConnt(x)}' for x in atoms]
        attrs = ['id1', 'id2', 'id3', 'id4']
        ids = [getattr(matches[0], x) for x in attrs if hasattr(matches[0], x)]
        nsmbl_cnnts = [
            f'{self.atoms[x].symbol}{self.atoms[x].conn}' for x in ids
        ]
        # C4~C4 84~88 replaced by C4.0~C4.0 86~88
        log_debug(
            f"{'~'.join(smbl_cnnts)} "
            f"{'~'.join(map(str, [x.GetIntProp(TYPE_ID) for x in atoms]))} "
            f"replaced by {'~'.join(map(str, nsmbl_cnnts))} {'~'.join(map(str, ids))}"
        )

    def getAngleAtoms(self, atom):
        """
        Get all three angle atoms from the input middle atom. The first atom has
        a TYPE_ID smaller than the third.

        :param atom 'rdkit.Chem.rdchem.Atom': the middle atom
        :return list of list: each sublist contains three atoms.
        """
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            return []
        neighbors = sorted(neighbors, key=lambda x: x.GetIntProp(TYPE_ID))
        return [[x, atom, y] for x, y in itertools.combinations(neighbors, 2)]

    def getMatchedAngles(self, atoms):
        """
        Get the matched angle force field types.

        :param atoms list of three 'rdkit.Chem.rdchem.Atom': atom for an angle
        :return list of 'oplsua.ANGLE': the matched parameters.
        """

        end_ids = [x.GetIntProp(ANGLE_ATM_ID) for x in atoms[::2]]
        if end_ids[0] > end_ids[1]:
            atoms = list(reversed(atoms))

        tids = tuple([x.GetIntProp(ANGLE_ATM_ID) for x in atoms])
        try:
            tids = ANGLE_ATOMS[tids]
        except KeyError:
            # C-OH (Tyr) is used as HO-C=O, needing CH2-COOH map as alpha-COOH bond
            pass
        try:
            return [self.angles[self.ang_map[tids[0], tids[1], tids[2]]]]
        except KeyError:
            pass
        msg = f"No exact params for angle between atom {', '.join(map(str, tids))}."
        log_debug(msg)

        partial_matches = [x for x in self.angles.values() if x.id2 == tids[1]]
        if not partial_matches:
            raise ValueError(
                f"No params for angle (middle atom type {tids[1]}).")
        matches = self.getMatchesFromEnds(atoms, partial_matches)
        if not matches:
            err = f"No params for angle between atom {', '.join(map(str, tids))}."
            raise ValueError(err)
        self.debugPrintReplacement(atoms, matches)

        return matches

    def getMatchesFromEnds(self, atoms, partial_matches):
        """
        Based on the symbols and connectivities of the two ends, filter the matches

        :param atoms 'rdkit.Chem.rdchem.Atom' list: atoms forming angle or dihedral
        :param partial_matches list of namedtuple: force field nametuple with
            the middle atom(s) matched.

        :return list of namedtuple: force field nametuples with ended atoms
            partial for fully matches.
        """
        eatoms = [atoms[0], atoms[-1]]
        o_symbols = set((x.GetSymbol(), self.getAtomConnt(x)) for x in eatoms)
        ff_atom_ids = [
            [x, x.id1, x.id4] if hasattr(x, 'id4') else [x, x.id1, x.id3]
            for x in partial_matches
        ]
        ff_symbols = {
            x[0]: set([(self.atoms[y].symbol, self.atoms[y].conn)
                       for y in x[1:]])
            for x in ff_atom_ids
        }
        # Both symbols and connectivities are matched
        matches = [x for x, y in ff_symbols.items() if y == o_symbols]

        if not matches:
            # Only symbols are matched
            o_symbols_partial = set(x[0] for x in o_symbols)
            matches = [
                x for x, y in ff_symbols.items()
                if set(z[0] for z in y) == o_symbols_partial
            ]
        return matches

    def getMatchedDihedrals(self, atoms):
        """
        Get the matched dihedral force field types.

        1) Exact match of all four atom types
        2) Exact match of torsion bond if found else forced match of the torsion
        3) End atom matching based on symbol and connectivity

        :param atoms list of three 'rdkit.Chem.rdchem.Atom': atom for a dihedral
        :return list of 'oplsua.DIHEDRAL': the matched parameters.
        """

        tids = [x.GetIntProp(DIHE_ATM_ID) for x in atoms]
        if tids[1] > tids[2]:
            # Flip the direction due to middle torsion atom id order
            tids = tids[::-1]
        match = self.dihe_map[tids[0], tids[1], tids[2], tids[3]]
        if match:
            return [self.dihedrals[match]]

        dihes = self.dihe_map[:, tids[1], tids[2], :]
        partial_matches = [self.dihedrals[x] for x in dihes[dihes != 0]]
        if not partial_matches:
            rpm_ids = DIHE_ATOMS[tuple(tids[1:3])]
            dihes = self.dihe_map[:, rpm_ids[0], rpm_ids[1], :]
            partial_matches = [self.dihedrals[x] for x in dihes[dihes != 0]]
        if not partial_matches:
            err = f"No params for dihedral (middle bonded atom types {tids[1]}~{tids[2]})."
            raise ValueError(err)
        matches = self.getMatchesFromEnds(atoms, partial_matches)
        if not matches:
            err = f"Cannot find params for dihedral between atom {'~'.join(map(str, tids))}."
            raise ValueError(err)
        return matches

    def molecular_weight(self, mol):
        """
        The molecular weight of one rdkit molecule.

        :parm mol: rdkit.Chem.rdchem.Mol one rdkit molecule.
        :return float: the total weight.
        """
        atypes = [x.GetIntProp(TYPE_ID) for x in mol.GetAtoms()]
        return round(sum(self.atoms[x].mass for x in atypes), 4)

    @property
    def improper_symbols(self):
        """
        Check and assert the current improper force field. These checks may be
        only good for this specific force field for even this specific file.
        """
        if self.symbol_impropers is not None:
            return self.symbol_impropers
        msg = "Impropers from the same symbols are of the same constants."
        # {1: 'CNCO', 2: 'CNCO', 3: 'CNCO' ...
        symbolss = {
            z:
            ''.join(
                [str(self.atoms[x.id3].conn)] +
                [self.atoms[y].symbol for y in [x.id1, x.id2, x.id3, x.id4]])
            for z, x in self.impropers.items()
        }
        # {'CNCO': (10.5, 180.0, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9), ...
        symbol_impropers = {}
        for id, symbols in symbolss.items():
            improper = self.impropers[id]
            if symbols not in symbol_impropers:
                symbol_impropers[symbols] = (
                    improper.ene,
                    improper.deg,
                    improper.n_parm,
                )
            assert symbol_impropers[symbols][:3] == (
                improper.ene,
                improper.deg,
                improper.n_parm,
            )
            symbol_impropers[symbols] += (improper.id, )
        log_debug(msg)

        # neighbors of CC(=O)C and CC(O)C have the same symbols
        msg = "Improper neighbor counts based on center conn and symbols are unique."
        # The third one is the center ('Improper Torsional Parameters' in prm)
        neighbors = [[x[0], x[3], x[1], x[2], x[4]]
                     for x in symbol_impropers.keys()]
        # The csmbls in getCountedSymbols is obtained from the following
        csmbls = sorted(set([y for x in neighbors for y in x[1:]]))  # CHNO
        counted = [self.countSymbols(x, csmbls=csmbls) for x in neighbors]
        assert len(symbol_impropers) == len(set(counted))
        log_debug(msg)
        self.symbol_impropers = {
            x: y[3:]
            for x, y in zip(counted, symbol_impropers.values())
        }
        return self.symbol_impropers

    @staticmethod
    def countSymbols(symbols, csmbls='CHNO'):
        """
        Count improper cluster symbols: the first is the center atom connectivity
        including implicit hydrogen atoms. The second is the center atom symbol,
        and the rest connects with the center.

        :param symbols list: the element symbols forming the improper cluster
            with first being the center
        :param csmbls str: all possible cluster symbols
        """
        # e.g., ['3', 'C', 'C', 'N', 'O']
        counted = [y + str(symbols[2:].count(y)) for y in csmbls]
        # e.g., '3CC1H0N1O1'
        return ''.join(symbols[:2] + counted)

    def getMatchedImpropers(self, atoms):
        """
        """
        symbols = [str(self.getAtomConnt(atoms[2])), atoms[2].GetSymbol()]
        symbols += [x.GetSymbol() for x in atoms[:2]]
        symbols += [atoms[3].GetSymbol()]
        counted = self.countSymbols(symbols)
        return self.improper_symbols[counted]
