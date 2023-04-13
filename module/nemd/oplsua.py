# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles opls-ua related typing, parameterization, assignment,
datafile, and in-script.
"""
import types
import base64
import chemparse
import itertools
import collections
import numpy as np
from rdkit import Chem
from collections import namedtuple

from nemd import symbols
from nemd import logutils
from nemd import fileutils
from nemd import constants
from nemd import environutils

BOND_ATM_ID = 'bond_atm_id'
ANGLE_ATM_ID = 'angle_atm_id'
DIHE_ATM_ID = 'dihe_atm_id'
RES_NUM = 'res_num'
IMPLICIT_H = 'implicit_h'
TYPE_ID = 'type_id'
LARGE_NUM = constants.LARGE_NUM

ATOM_TYPE = namedtuple('ATOM_TYPE', [
    'id', 'formula', 'symbol', 'description', 'atomic_number', 'mass', 'conn'
])
VDW = namedtuple('VDW', ['id', 'dist', 'ene'])
BOND = namedtuple('BOND', ['id', 'id1', 'id2', 'dist', 'ene'])
ANGLE = namedtuple('ANGLE', ['id', 'id1', 'id2', 'id3', 'ene', 'angle'])
UREY_BRADLEY = namedtuple('UREY_BRADLEY', ['id1', 'id2', 'id3', 'ene', 'dist'])
IMPROPER = namedtuple(
    'IMPROPER', ['id', 'id1', 'id2', 'id3', 'id4', 'ene', 'angle', 'n_parm'])
ENE_ANG_N = namedtuple('ENE_ANG_N', ['ene', 'angle', 'n_parm'])
DIHEDRAL = namedtuple('DIHEDRAL',
                      ['id', 'id1', 'id2', 'id3', 'id4', 'constants'])

UA = namedtuple('UA', ['sml', 'mp', 'hs', 'dsc'])

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


def get_opls_parser():
    """
    Read and parser opls force field file.
    :return 'OplsParser': the parser with force field information
    """
    opls_parser = OplsParser()
    opls_parser.read()
    return opls_parser


class OplsTyper:
    """
    Type the atoms and map SMILES fragments.
    """

    OPLSUA = 'OPLSUA'
    TYPE_ID = TYPE_ID
    RES_NUM = RES_NUM
    BOND_ATM_ID = BOND_ATM_ID
    ANGLE_ATM_ID = ANGLE_ATM_ID
    DIHE_ATM_ID = DIHE_ATM_ID
    IMPLICIT_H = IMPLICIT_H
    LARGE_NUM = constants.LARGE_NUM

    # yapf: disable
    SMILES = [UA(sml='[Li+]', mp=(197,), hs=None, dsc='Li+ Lithium Ion'),
              UA(sml='[Na+]', mp=(198,), hs=None, dsc='Na+ Sodium Ion'),
              UA(sml='[K+]', mp=(199,), hs=None, dsc='K+ Potassium Ion'),
              UA(sml='[Rb+]', mp=(200,), hs=None, dsc='Rb+ Rubidium Ion'),
              UA(sml='[Cs+]', mp=(201,), hs=None, dsc='Cs+ Cesium Ion'),
              UA(sml='[Mg+]', mp=(202,), hs=None, dsc='Mg+2 Magnesium Ion'),
              UA(sml='[Ca+]', mp=(203,), hs=None, dsc='Ca+2 Calcium Ion'),
              UA(sml='[Sr+]', mp=(204,), hs=None, dsc='Sr+2 Strontium Ion'),
              UA(sml='[Ba+]', mp=(205,), hs=None, dsc='Ba+2 Barium Ion'),
              UA(sml='[F-]', mp=(206,), hs=None, dsc='F- Fluoride Ion'),
              UA(sml='[Cl-]', mp=(207,), hs=None, dsc='Cl- Chloride Ion'),
              UA(sml='[Br-]', mp=(208,), hs=None, dsc='Br- Bromide Ion'),
              UA(sml='[He]', mp=(209,), hs=None, dsc='Helium Atom'),
              UA(sml='[Ne]', mp=(210,), hs=None, dsc='Neon Atom'),
              UA(sml='[Ar]', mp=(211,), hs=None, dsc='Argon Atom'),
              UA(sml='[Kr]', mp=(212,), hs=None, dsc='Krypton Atom'),
              UA(sml='[Xe]', mp=(213,), hs=None, dsc='Xenon Atom'),
              UA(sml='O', mp=(77,), hs={77: 78}, dsc='Water (TIP3P)'),
              UA(sml='C', mp=(81, ), hs=None, dsc='CH4 Methane'),
              UA(sml='CC', mp=(82, 82,), hs=None, dsc='Ethane'),
              UA(sml='CO', mp=(106, 104,), hs={104:105}, dsc='Methanol'),
              UA(sml='CCO', mp=(83, 107, 104,), hs={104: 105}, dsc='Ethanol'),
              UA(sml='CCC', mp=(83, 86, 83,), hs=None, dsc='Propane'),
              UA(sml='CCCC', mp=(83, 86, 86, 83,), hs=None, dsc='n-Butane'),
              UA(sml='CC(C)C', mp=(84, 88, 84, 84, ), hs=None, dsc='Isobutane'),
              UA(sml='CC(C)O', mp=(84, 108, 84, 104,), hs={104:105}, dsc='Isopropanol'),
              UA(sml='CC=CC', mp=(84, 89, 89, 84, ), hs=None, dsc='2-Butene'),
              UA(sml='CN(C)C=O', mp=(156, 148, 156, 153, 151,), hs=None, dsc='N,N-Dimethylformamide'),
              # "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
              UA(sml='O=CO', mp=(134, 133, 135), hs={135: 136}, dsc='Carboxylic Acid'),
              # "Methyl", "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
              UA(sml='CC(=O)O', mp=(137, 133, 134, 135), hs={135: 136}, dsc='Ethanoic acid')]

    SMILES = list(reversed(SMILES))
    ATOM_TOTAL = {i: i for i in range(1, 214)}
    BOND_ATOM = ATOM_TOTAL.copy()
    # "O Peptide Amide" "COH (zeta) Tyr" "OH Tyr"  "H(O) Ser/Thr/Tyr"
    BOND_ATOM.update({134: 2, 133: 26, 135: 23, 136: 24, 153: 72, 148: 3, 108: 107})
    ANGLE_ATOM = ATOM_TOTAL.copy()
    ANGLE_ATOM.update({134: 2, 133: 17, 135: 76, 136: 24, 148: 3, 153: 72, 108: 107})
    DIHE_ATOM = ATOM_TOTAL.copy()
    DIHE_ATOM.update({134: 11, 133: 26, 135: 76, 136: 24, 148: 3, 153: 72, 108: 107})
    # C-OH (Tyr) is used as HO-C=O, which needs CH2-COOH map as alpha-COOH bond
    BOND_ATOMS = {(26, 86): [16, 17], (26, 88): [16, 17], (86, 107): [86, 86]}
    ANGLE_ATOMS = {(84, 107, 84): (86, 88, 86), (84, 107, 86): (86, 88, 83)}
    DIHE_ATOMS = {(26,86,): (1,6,), (26,88,): (1,6,), (88, 107,): (6, 22,), (86, 107,): (6, 25,)}
    # yapf: enable
    # https://docs.lammps.org/Howto_tip3p.html
    TIP3P = 'TIP3P'
    SPC = 'SPC'
    WATER_TIP3P = f'Water ({TIP3P})'
    WATER_SPC = f'Water ({SPC})'
    UA_WATER_SPC = UA(sml='O', mp=(79,), hs={79: 80}, dsc=WATER_SPC)
    # yapf: enable

    WMODELS = [TIP3P, SPC]
    FF_MODEL = {OPLSUA: WMODELS}
    OPLSUA_TIP3P = f'{OPLSUA},{TIP3P}'

    def __init__(self, mol, wmodel=TIP3P):
        """
        :param mol 'rdkit.Chem.rdchem.Mol': molecule to assign FF types
        :param wmodel str: the model type for water
        """
        self.mol = mol
        if wmodel == self.TIP3P:
            return
        idx = [
            x for x, y in enumerate(self.SMILES) if y.dsc == self.WATER_TIP3P
        ][0]
        self.SMILES[idx] = self.UA_WATER_SPC

    def run(self):
        """
        Assign atom types for force field assignment.
        """
        marked_smiles = {}
        marked_atom_ids = []
        res_num = 1
        for sml in self.SMILES:
            frag = Chem.MolFromSmiles(sml.sml)
            matches = self.mol.GetSubstructMatches(frag,
                                                   maxMatches=self.LARGE_NUM)
            matches = [self.filterMatch(x, frag) for x in matches]
            res_num, matom_ids = self.markMatches(matches, sml, res_num)
            if not matom_ids:
                continue
            cnt = collections.Counter([len(x) for x in matom_ids])
            cnt_exp = str(len(matom_ids)) + ' matches ' + ','.join(
                [f'{x}*{y}' for x, y in cnt.items()])
            marked_smiles[sml.sml] = cnt_exp
            marked_atom_ids += [y for x in matom_ids for y in x]
            if all(x.HasProp(self.TYPE_ID) for x in self.mol.GetAtoms()):
                break
        log_debug(
            f"{len(marked_atom_ids)} out of {self.mol.GetNumAtoms()} atoms marked"
        )
        log_debug(f"{res_num - 1} residues found.")
        [log_debug(f'{x}: {y}') for x, y in marked_smiles.items()]

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
                atom.GetIntProp(self.TYPE_ID)
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
        atom.SetIntProp(self.TYPE_ID, type_id)
        atom.SetIntProp(self.RES_NUM, res_num)
        # BOND_ATM_ID defines bonding parameters
        atom.SetIntProp(self.BOND_ATM_ID, self.BOND_ATOM[type_id])
        atom.SetIntProp(self.ANGLE_ATM_ID, self.ANGLE_ATOM[type_id])
        atom.SetIntProp(self.DIHE_ATM_ID, self.DIHE_ATOM[type_id])


class OplsParser:
    """
    Parse force field file and map atomic details.
    """

    FILE_PATH = fileutils.get_ff(name=fileutils.OPLSUA, ext=fileutils.RRM_EXT)

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

    BOND_ATM_ID = BOND_ATM_ID
    ANGLE_ATM_ID = ANGLE_ATM_ID
    DIHE_ATM_ID = DIHE_ATM_ID
    IMPLICIT_H = IMPLICIT_H
    TYPE_ID = TYPE_ID

    def __init__(self, filepath=None):
        """
        :param filepath str: the path to the force field file.
        """
        self.filepath = filepath
        if self.filepath is None:
            self.filepath = self.FILE_PATH
        self.raw_content = {}
        self.atoms = {}
        self.vdws = {}
        self.bonds = {}
        self.angles = {}
        self.urey_bradleys = {}
        self.impropers = {}
        self.dihedrals = {}
        self.charges = {}

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
        for id, line in enumerate(self.raw_content[self.BOND_MK], 1):
            # 'bond        104  107          386.00     1.4250'
            _, id1, id2, ene, dist = line.split()
            self.bonds[id] = BOND(id=id,
                                  id1=int(id1),
                                  id2=int(id2),
                                  ene=float(ene),
                                  dist=float(dist))

    def setAngle(self):
        """
        Set angle parameters based on 'Angle Bending Parameters' block.
        """
        for id, line in enumerate(self.raw_content[self.ANGLE_MK], 1):
            # 'angle        83  107  104      80.00     109.50'
            _, id1, id2, id3, ene, angle = line.split()
            self.angles[id] = ANGLE(id=id,
                                    id1=int(id1),
                                    id2=int(id2),
                                    id3=int(id3),
                                    ene=float(ene),
                                    angle=float(angle))

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
                                          angle=float(angle),
                                          n_parm=int(n_parm))

    def setDihedral(self):
        """
        Set dihedral parameters based on 'Torsional Parameters' block.
        """
        for id, line in enumerate(self.raw_content[self.TORSIONAL_MK], 1):
            # torsion       2    1    3    4            0.650    0.0  1      2.500  180.0  2
            line_splitted = line.split()
            ids, enes = line_splitted[1:5], line_splitted[5:]
            ene_ang_ns = tuple(
                ENE_ANG_N(ene=float(x), angle=float(y), n_parm=int(z))
                for x, y, z in zip(enes[::3], enes[1::3], enes[2::3]))
            self.dihedrals[id] = DIHEDRAL(id=id,
                                          id1=int(ids[0]),
                                          id2=int(ids[1]),
                                          id3=int(ids[2]),
                                          id4=int(ids[3]),
                                          constants=ene_ang_ns)

    def getMatchedBonds(self, bonded_atoms):
        """
        Get force field matched bonds. The searching and approximation follows:

        1) Forced mapping via BOND_ATOMS to connect force field fragments.
        2) Exact match for current atom types.
        3) Matching of one atom with the other's symbol and connectivity matched
        4) Matching of one atom with only the other's symbol matched

        :raise ValueError: If the above failed

        :param bonded_atoms: list of two bonded atoms sorted by BOND_ATM_ID
        :return list of 'oplsua.BOND': bond information
        """

        atypes = [x.GetIntProp(self.BOND_ATM_ID) for x in bonded_atoms]
        try:
            atypes = OplsTyper.BOND_ATOMS[tuple(sorted(atypes))]
        except KeyError:
            # C-OH (Tyr) is used as HO-C=O, needing CH2-COOH map as alpha-COOH bond
            pass
        # Exact match between two atom type ids
        matches = [x for x in self.bonds.values() if [x.id1, x.id2] == atypes]
        if matches:
            return matches

        msg = f"No exact params for bond between atom type {atypes[0]} and {atypes[1]}."
        log_debug(msg)
        bond_score, type_set = {}, set(atypes)
        for bond in self.bonds.values():
            matched = type_set.intersection([bond.id1, bond.id2])
            if len(matched) != 1:
                continue
            # Compare the unmatched and sore them
            atom_id = set([bond.id1, bond.id2]).difference(matched).pop()
            atom = [
                x for x in bonded_atoms
                if x.GetIntProp(self.BOND_ATM_ID) not in [bond.id1, bond.id2]
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

    @classmethod
    def getAtomConnt(cls, atom):
        """
        Get the atomic connectivity information.

        :param atom 'rdkit.Chem.rdchem.Atom': the connectivity of this atom
        :return int: the number of bonds connected to this atom including the
            implicit hydrogen.
        """

        implicit_h_num = atom.GetIntProp(cls.IMPLICIT_H) if atom.HasProp(
            cls.IMPLICIT_H) else 0
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
            f"{'~'.join(map(str, [x.GetIntProp(self.TYPE_ID) for x in atoms]))} "
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
        neighbors = sorted(neighbors, key=lambda x: x.GetIntProp(self.TYPE_ID))
        return [[x, atom, y] for x, y in itertools.combinations(neighbors, 2)]

    def getMatchedAngles(self, atoms):
        """
        Get the matched angle force field types.

        :param atoms list of three 'rdkit.Chem.rdchem.Atom': atom for an angle
        :return list of 'oplsua.ANGLE': the matched parameters.
        """

        end_ids = [x.GetIntProp(self.ANGLE_ATM_ID) for x in atoms[::2]]
        if end_ids[0] > end_ids[1]:
            atoms = list(reversed(atoms))

        tids = tuple([x.GetIntProp(self.ANGLE_ATM_ID) for x in atoms])
        try:
            tids = OplsTyper.ANGLE_ATOMS[tids]
        except KeyError:
            # C-OH (Tyr) is used as HO-C=O, needing CH2-COOH map as alpha-COOH bond
            pass
        matches = [
            x for x in self.angles.values()
            if tids == tuple([x.id1, x.id2, x.id3])
        ]
        if matches:
            return matches
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
            x[0]:
            set([(self.atoms[y].symbol, self.atoms[y].conn) for y in x[1:]])
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

        tids = [x.GetIntProp(self.DIHE_ATM_ID) for x in atoms]
        if tids[1] > tids[2]:
            # Flip the direction due to middle torsion atom id order
            tids = tids[::-1]
        matches = [
            x for x in self.dihedrals.values()
            if tids == [x.id1, x.id2, x.id3, x.id4]
        ]
        if matches:
            return matches

        partial_matches = [
            x for x in self.dihedrals.values()
            if x.id2 == tids[1] and x.id3 == tids[2]
        ]
        if not partial_matches:
            rpm_ids = OplsTyper.DIHE_ATOMS[tuple(tids[1:3])]
            partial_matches = [
                x for x in self.dihedrals.values()
                if set([x.id2, x.id3]) == set(rpm_ids)
            ]
        if not partial_matches:
            err = f"No params for dihedral (middle bonded atom types {tids[1]}~{tids[2]})."
            raise ValueError(err)
        matches = self.getMatchesFromEnds(atoms, partial_matches)
        if not matches:
            err = f"Cannot find params for dihedral between atom {'~'.join(map(str, tids))}."
            raise ValueError(err)
        return matches


class LammpsIn(fileutils.LammpsInput):
    """
    Class to write out LAMMPS in script.
    """

    IN_EXT = '.in'
    DATA_EXT = '.data'

    LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
    LJ_CUT = 'lj/cut'
    GEOMETRIC = 'geometric'
    ARITHMETIC = 'arithmetic'
    SIXTHPOWER = 'sixthpower'

    MIX = 'mix'
    PPPM = 'pppm'
    REAL = 'real'
    FULL = 'full'
    OPLS = 'opls'
    CVFF = 'cvff'
    FIRE = 'fire'
    MIN_STYLE = 'min_style'
    HARMONIC = 'harmonic'
    LJ_COUL = 'lj/coul'

    def __init__(self,
                 jobname,
                 lj_cut=11.,
                 coul_cut=11.,
                 timestep=1,
                 concise=True):
        """
        :param jobname str: jobname based on which out filenames are defined
        :param lj_cut float: cut off distance for Lennard-Jones potential
        :param coul_cut float: cut off distance for Coulombic pairwise interaction
        :param timestep float: timestep size for subsequent molecular dynamics
            simulations
        """
        self.jobname = jobname
        self.lj_cut = lj_cut
        self.coul_cut = coul_cut
        self.timestep = timestep
        self.concise = concise
        self.lammps_in = self.jobname + self.IN_EXT
        self.lammps_data = self.jobname + self.DATA_EXT
        self.units = self.REAL
        self.atom_style = self.FULL
        self.bond_style = self.HARMONIC
        self.angle_style = self.HARMONIC
        self.dihedral_style = self.OPLS
        self.improper_style = self.CVFF
        self.pair_style = {
            self.LJ_CUT_COUL_LONG:
            f"{self.LJ_CUT_COUL_LONG} {self.lj_cut} {self.coul_cut}",
            self.LJ_CUT: f"{self.LJ_CUT} {self.lj_cut}"
        }
        self.in_fh = None
        self.is_debug = environutils.is_debug()

    def writeLammpsIn(self):
        """
        Write out LAMMPS in script.
        """
        with open(self.lammps_in, 'w') as self.in_fh:
            self.writeDescriptions()
            self.readData()
            self.writeMinimize()
            self.writeTimestep()
            self.writeRun()

    def writeDescriptions(self):
        """
        Write in script description section.
        """
        self.in_fh.write(f"{self.UNITS} {self.units}\n")
        self.in_fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
        self.in_fh.write(f"{self.BOND_STYLE} {self.bond_style}\n")
        self.in_fh.write(f"{self.ANGLE_STYLE} {self.angle_style}\n")
        self.in_fh.write(f"{self.DIHEDRAL_STYLE} {self.dihedral_style}\n")
        self.in_fh.write(f"{self.IMPROPER_STYLE} {self.improper_style}\n")
        pair_style = self.LJ_CUT_COUL_LONG if self.hasCharge() else self.LJ_CUT
        self.in_fh.write(f"{self.PAIR_STYLE} {self.pair_style[pair_style]}\n")
        self.in_fh.write(f"{self.PAIR_MODIFY} {self.MIX} {self.GEOMETRIC}\n")
        self.in_fh.write(f"{self.SPECIAL_BONDS} {self.LJ_COUL} 0 0 0.5 \n")
        if self.hasCharge():
            self.in_fh.write(f"{self.KSPACE_STYLE} {self.PPPM} 0.0001\n")
        self.in_fh.write(f"log log.{self.jobname}\n")

    def hasCharge(self):
        """
        Whether any atom has non-zero charge. This method should be overwritten
        when force field and structure are available.

        :return bool: True if any atom has non-zero charge.
        """

        return True

    def readData(self):
        """
        Write data file related information.
        """
        self.in_fh.write(f"{self.READ_DATA} {self.lammps_data}\n")

    def writeMinimize(self, min_style=FIRE, dump=True):
        """
        Write commands related to minimization.

        :param min_style str: cg, fire, spin, etc.
        :param dump bool: Whether dump out trajectory.
        """
        if dump:
            self.in_fh.write(
                "dump 1 all custom 1000 dump.custom id xu yu zu\n")
            self.in_fh.write("dump_modify 1 sort id\n")
        self.in_fh.write(f"{self.MIN_STYLE} {min_style}\n")
        self.in_fh.write("minimize 1.0e-6 1.0e-8 10000 100000\n")

    def writeTimestep(self):
        """
        Write commands related to timestep.
        """
        self.in_fh.write(f'timestep {self.timestep}\n')
        self.in_fh.write('thermo_modify flush yes\n')
        self.in_fh.write('thermo 1000\n')

    def dumpImproper(self):
        """
        Compute and dump improper values with type.
        """
        self.in_fh.write(
            'compute 1 all property/local itype iatom1 iatom2 iatom3 iatom4\n')
        self.in_fh.write('compute 2 all improper/local chi\n')
        self.in_fh.write('dump 1i all local 1000 tmp.dump index c_1[1] c_2\n')

    def writeRun(self,
                 start_temp=10.,
                 target_temp=None,
                 tdamp=100,
                 pdamp=1000,
                 nve_only=False,
                 npt=True):
        """
        Write command to further equilibrate the system.

        :param start_temp float: the starting temperature for relaxation
        :param target_temp float: the target temperature at the end of the relaxation
        :param tdamp float: damping factor to control temperature
        :param pdamp float: damping factor to control pressure
        :param nve_only bool: only run nve ensemble (one single small molecule)
        :param npt bool: run npt ensemble (molecule can fill the space)
        """

        self.in_fh.write(f"velocity all create {start_temp} 482748\n")
        if nve_only:
            # NVT on single molecule gives nan coords (guess due to translation)
            self.in_fh.write("fix 1 all nve\n")
            self.in_fh.write("run 2000\n")
            return

        self.in_fh.write(
            f"fix 1 all nvt temp {start_temp} {start_temp} {self.timestep * tdamp}\n"
        )
        self.in_fh.write("run 20000\n")

        if not npt:
            return

        # Run relaxation on systems of multiple molecules
        self.in_fh.write("unfix 1\n")
        self.in_fh.write(
            f"fix 1 all npt temp {start_temp} {start_temp} {self.timestep * tdamp} "
            f"iso 1 1 {self.timestep * pdamp}\n")
        self.in_fh.write("run 10000\n")

        if target_temp is None:
            return

        self.in_fh.write("unfix 1\n")
        self.in_fh.write(
            f"fix 1 all npt temp {start_temp} {target_temp} {self.timestep * tdamp} "
            f"iso 1 1 {self.timestep * pdamp}\n")
        self.in_fh.write("run 100000\n")

        self.in_fh.write("unfix 1\n")
        self.in_fh.write(
            f"fix 1 all npt temp {target_temp} {target_temp} {self.timestep * tdamp} "
            f"iso 1 1 {self.timestep * pdamp}\n")
        self.in_fh.write("run 100000\n")


class LammpsData(LammpsIn):
    """
    Class to write out LAMMPS data file.
    """

    LAMMPS_DESCRIPTION = 'LAMMPS Description'

    TYPE_ID = TYPE_ID
    IMPLICIT_H = IMPLICIT_H
    ATOM_ID = 'atom_id'
    RES_NUM = RES_NUM
    NEIGHBOR_CHARGE = 'neighbor_charge'
    BOND_ATM_ID = OplsTyper.BOND_ATM_ID

    ATOMS = 'atoms'
    BONDS = 'bonds'
    ANGLES = 'angles'
    DIHEDRALS = 'dihedrals'
    IMPROPERS = 'impropers'
    STRUCT_DSP = [ATOMS, BONDS, ANGLES, DIHEDRALS, IMPROPERS]

    ATOM_TYPES = 'atom types'
    BOND_TYPES = 'bond types'
    ANGLE_TYPES = 'angle types'
    DIHEDRAL_TYPES = 'dihedral types'
    IMPROPER_TYPES = 'improper types'
    TYPE_DSP = [
        ATOM_TYPES, BOND_TYPES, ANGLE_TYPES, DIHEDRAL_TYPES, IMPROPER_TYPES
    ]

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    BOX_DSP = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]
    BUFFER = [4., 4., 4.] # yapf: disable

    MASSES = 'Masses'
    PAIR_COEFFS = 'Pair Coeffs'
    BOND_COEFFS = 'Bond Coeffs'
    ANGLE_COEFFS = 'Angle Coeffs'
    DIHEDRAL_COEFFS = 'Dihedral Coeffs'
    IMPROPER_COEFFS = 'Improper Coeffs'
    ATOMS_CAP = ATOMS.capitalize()
    BONDS_CAP = BONDS.capitalize()
    ANGLES_CAP = ANGLES.capitalize()
    DIHEDRALS_CAP = DIHEDRALS.capitalize()
    IMPROPERS_CAP = IMPROPERS.capitalize()

    MARKERS = [
        MASSES, PAIR_COEFFS, BOND_COEFFS, ANGLE_COEFFS, DIHEDRAL_COEFFS,
        IMPROPER_COEFFS, ATOMS_CAP, BONDS_CAP, ANGLES_CAP, DIHEDRALS_CAP,
        IMPROPERS_CAP
    ]

    IMPROPER_CENTER_SYMBOLS = symbols.CARBON + symbols.HYDROGEN

    def __init__(self,
                 mols,
                 ff,
                 jobname,
                 *arg,
                 concise=True,
                 box=None,
                 **kwarg):
        """
        :param mols dict: keys are the molecule ids, and values are
            'rdkit.Chem.rdchem.Mol'
        :param ff 'oplsua.OplsParser': the force field information
        :param jobname str: jobname based on which out filenames are defined
        :param concise bool: If False, all the atoms in the force field file
            shows up in the force field section of the data file. If True, only
            the present ones are writen into the data file.
        :param concise bool: the force field information
        :param box list: the PBC limits (xlo, xhi, ylo, yhi, zlo, zhi)
        """
        super().__init__(jobname, *arg, **kwarg)
        self.ff = ff
        self.mols = mols
        self.jobname = jobname
        self.concise = concise
        self.box = box
        self.atoms = {}
        self.bonds = {}
        self.rvrs_bonds = {}
        self.rvrs_angles = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.symbol_impropers = {}
        self.atm_types = []
        self.bnd_types = []
        self.ang_types = []
        self.dihe_types = []
        self.impr_types = []
        self.data_fh = None

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        charges = [
            self.ff.charges[x.GetIntProp(self.TYPE_ID)] for x in self.atom
        ]
        return any(charges)

    def writeDumpModify(self):
        """
        Write dump modify commands so that dump command can write out element.
        """
        atoms = self.ff.atoms.values()
        if self.concise:
            atoms = [x for x in atoms if x.id in self.atm_types]
        smbs = ' '.join(map(str, [x.symbol for x in atoms]))
        self.in_fh.write(f"dump_modify 1 element {smbs}\n")

    def writeRun(self, *arg, **kwarg):
        """
        Write command to further equilibrate the system.
        """
        nve_only = len(self.mols) == 1 and self.mols[1].GetNumAtoms() < 10
        npt = len(self.mols) > 1
        super().writeRun(*arg, nve_only=nve_only, npt=npt, **kwarg)

    def writeData(self, adjust_coords=True):
        """
        Write out LAMMPS data file.

        :param adjust_coords bool: whether adjust coordinates of the molecules.
            This only good for a small piece as clashes between non-bonded atoms
            may be introduced.
        """

        with open(self.lammps_data, 'w') as self.data_fh:
            self.setAtoms()
            self.balanceCharge()
            self.setBonds()
            self.adjustBondLength(adjust_coords)
            self.setAngles()
            self.setDihedrals()
            self.setImproperSymbols()
            self.setImpropers()
            self.removeAngles()
            self.removeUnused()
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writePairCoeffs()
            self.writeBondCoeffs()
            self.writeAngleCoeffs()
            self.writeDihedralCoeffs()
            self.writeImproperCoeffs()
            self.writeAtoms()
            self.writeBonds()
            self.writeAngles()
            self.writeDihedrals()
            self.writeImpropers()

    def setAtoms(self):
        """
        Set atom property.
        """

        # atom id is stored as per atom property instead of global dict
        for atom_id, atom in enumerate(self.atom, start=1):
            atom.SetIntProp(self.ATOM_ID, atom_id)

    @property
    def atom(self):
        """
        Handy way to get all atoms.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return (atom for mol in self.molecule for atom in mol.GetAtoms())

    @property
    def molecule(self):
        """
        Handy way to get all molecules.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return [mol for mol in self.mols.values()]

    def balanceCharge(self):
        """
        Balance the charge when residues are not neutral.
        """

        for mol in self.mols.values():
            res_charge = collections.defaultdict(float)
            for atom in mol.GetAtoms():
                res_num = atom.GetIntProp(self.RES_NUM)
                type_id = atom.GetIntProp(self.TYPE_ID)
                res_charge[res_num] += self.ff.charges[type_id]

            for bond in mol.GetBonds():
                batom, eatom = bond.GetBeginAtom(), bond.GetEndAtom()
                bres_num = batom.GetIntProp(self.RES_NUM)
                eres_num = eatom.GetIntProp(self.RES_NUM)
                if bres_num == eres_num:
                    continue
                for atom, natom in [[batom, eatom], [eatom, batom]]:
                    try:
                        # When the atom has multiple residue neighbors
                        charge = atom.GetDoubleProp(self.NEIGHBOR_CHARGE)
                    except KeyError:
                        charge = 0.0
                    ncharge = res_charge[natom.GetIntProp(self.RES_NUM)]
                    atom.SetDoubleProp(self.NEIGHBOR_CHARGE, charge - ncharge)

    def setBonds(self):
        """
        Set bonding information.
        """
        bonds = [bond for mol in self.molecule for bond in mol.GetBonds()]
        for bond_id, bond in enumerate(bonds, start=1):
            bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            # BOND_ATM_ID defines bonding parameters marked during atom typing
            bonded_atoms = sorted(bonded_atoms,
                                  key=lambda x: x.GetIntProp(self.BOND_ATM_ID))
            matches = self.ff.getMatchedBonds(bonded_atoms)
            bond = matches[0]
            atom_id1 = bonded_atoms[0].GetIntProp(self.ATOM_ID)
            atom_id2 = bonded_atoms[1].GetIntProp(self.ATOM_ID)
            atom_ids = sorted([atom_id1, atom_id2])
            self.bonds[bond_id] = (
                bond.id,
                *atom_ids,
            )
            self.rvrs_bonds[tuple(atom_ids)] = bond.id

    def adjustBondLength(self, adjust_bond_legnth=True):
        """
        Adjust bond length according to the force field parameters.

        :param adjust_bond_legnth bool: adjust bond length if True.
        """
        if not adjust_bond_legnth:
            return

        for mol in self.molecule:
            conformer = mol.GetConformer()
            for bond in mol.GetBonds():
                bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
                ids = set([x.GetIntProp(self.ATOM_ID) for x in bonded_atoms])
                bond_type = self.rvrs_bonds[tuple(sorted(ids))]
                dist = self.ff.bonds[bond_type].dist
                idxs = [x.GetIdx() for x in bonded_atoms]
                Chem.rdMolTransforms.SetBondLength(conformer, *idxs, dist)

    def adjustCoords(self):
        """
        Adjust the coordinates based bond length etc.
        """
        self.setAtoms()
        self.setBonds()
        self.adjustBondLength()

    def setAngles(self):
        """
        Set angle force field matches.
        """

        angle_atoms = (y for x in self.atom for y in self.ff.getAngleAtoms(x))
        for angle_id, atoms in enumerate(angle_atoms, start=1):
            angle = self.ff.getMatchedAngles(atoms)[0]
            atom_ids = tuple(x.GetIntProp(self.ATOM_ID) for x in atoms)
            self.angles[angle_id] = (angle.id, ) + atom_ids
            self.rvrs_angles[tuple(atom_ids)] = angle_id

    def setDihedrals(self):
        """
        Set the dihedral angles of the molecules.
        """

        dihe_atoms = self.getDiheAtoms()
        for dihedral_id, atoms in enumerate(dihe_atoms, start=1):
            dihedral = self.ff.getMatchedDihedrals(atoms)[0]
            atom_ids = tuple([x.GetIntProp(self.ATOM_ID) for x in atoms])
            self.dihedrals[dihedral_id] = (dihedral.id, ) + atom_ids

    def getDiheAtoms(self):
        """
        Get the dihedral atoms of all molecules.

        :return list of list: each sublist has four atoms forming a dihedral angle.
        """
        return [y for x in self.molecule for y in self.getDihAtomsFromMol(x)]

    def getDihAtomsFromMol(self, mol):
        """
        Get the dihedral atoms of this molecule.

        NOTE: Flipping the order the four dihedral atoms yields the same dihedral,
        and only one of them is returned.

        :param 'rdkit.Chem.rdchem.Mol': the molecule to get dihedral atoms.
        :return list of list: each sublist has four atom ids forming a dihedral angle.
        """
        atomss = [y for x in mol.GetAtoms() for y in self.getDihedralAtoms(x)]
        # 1-2-3-4 and 4-3-2-1 are the same dihedral
        atomss_no_flip = []
        atom_idss = set()
        for atoms in atomss:
            atom_ids = tuple(x.GetIdx() for x in atoms)
            if atom_ids in atom_idss:
                continue
            atom_idss.add(atom_ids)
            atom_idss.add(atom_ids[::-1])
            atomss_no_flip.append(atoms)
        return atomss_no_flip

    def getDihedralAtoms(self, atom):
        """
        Get the dihedral atoms whose torsion bonded atoms contain this atom.

        :param atom 'rdkit.Chem.rdchem.Atom': the middle atom of the dihedral
        :return list of list: each sublist has four atom ids forming a dihedral
            angle.
        """
        dihe_atoms = []
        atomss = self.ff.getAngleAtoms(atom)
        atomss += [x[::-1] for x in atomss]
        for satom, matom, eatom in atomss:
            presented = set([matom.GetIdx(), eatom.GetIdx()])
            dihe_4ths = [
                y for x in self.ff.getAngleAtoms(eatom) for y in x
                if y.GetIdx() not in presented
            ]
            for dihe_4th in dihe_4ths:
                dihe_atoms.append([satom, matom, eatom, dihe_4th])

        return dihe_atoms

    def setImproperSymbols(self):
        """
        Check and assert the current improper force field. These checks may be
        only good for this specific force field for even this specific file.
        """
        msg = "Impropers from the same symbols are of the same constants."
        # {1: 'CNCO', 2: 'CNCO', 3: 'CNCO' ...
        symbolss = {
            z: ''.join([
                self.ff.atoms[y].symbol for y in [x.id1, x.id2, x.id3, x.id4]
            ])
            for z, x in self.ff.impropers.items()
        }
        # {'CNCO': (10.5, 180.0, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9), ...
        symbol_impropers = {}
        for id, symbols in symbolss.items():
            improper = self.ff.impropers[id]
            if symbols not in symbol_impropers:
                symbol_impropers[symbols] = (
                    improper.ene,
                    improper.angle,
                    improper.n_parm,
                )
            assert symbol_impropers[symbols][:3] == (
                improper.ene,
                improper.angle,
                improper.n_parm,
            )
            symbol_impropers[symbols] += (improper.id, )
        log_debug(msg)

        msg = "Improper neighbor counts based on symbols are unique."
        # The third one is the center ('Improper Torsional Parameters' in prm)
        neighbors = [[x[2], x[0], x[1], x[3]] for x in symbol_impropers.keys()]
        # The csmbls in getCountedSymbols is obtained from the following
        csmbls = sorted(set([y for x in neighbors[1:] for y in x]))  # CHNO
        counted = [self.countSymbols(x, csmbls=csmbls) for x in neighbors]
        assert len(symbol_impropers) == len(set(counted))
        log_debug(msg)

        self.symbol_impropers = {
            x: y[3:]
            for x, y in zip(counted, symbol_impropers.values())
        }

    @staticmethod
    def countSymbols(symbols, csmbls='CHNO'):
        """
        Count improper cluster symbols: the first is the center atoms and the
        rest connects with the center.

        :param symbols list: the element symbols forming the improper cluster
            with first being the center
        :param csmbls str: all possible cluster symbols
        """
        # e.g., ['C1', 'H0', 'N1', 'O1']
        counted = [y + str(symbols[1:].count(y)) for y in csmbls]
        # e.g., 'CC1H0N1O1'
        return ''.join([symbols[0]] + counted)

    def setImpropers(self, csymbols=IMPROPER_CENTER_SYMBOLS):
        """
        Set improper angles based on center atoms and neighbor symbols.

        :param csymbols str: each Char is one possible center element

        In short:
        1) sp2 sites and united atom CH groups (sp3 carbons) needs improper
         (though I saw a reference using improper for sp3 N)
        2) No rules for a center atom. (Charmm asks order for symmetricity)
        3) Number of internal geometry variables (3N_atom – 6) deletes one angle

        The details are the following:

        When the Weiner et al. (1984,1986) force field was developed, improper
        torsions were designated for specific sp2 sites, as well as for united
        atom CH groups - sp3 carbons with one implicit hydrogen.
        Ref: http://ambermd.org/Questions/improp.html

        There are no rules for a center atom. You simply define two planes, each
        defined by three atoms. The angle is given by the angle between these
        two planes. (from hess)
        ref: https://gromacs.bioexcel.eu/t/the-atom-order-i-j-k-l-in-defining-an
        -improper-dihedral-in-gromacs-using-the-opls-aa-force-field/3658

        The CHARMM convention in the definition of improper torsion angles is to
        list the central atom in the first position, while no rule exists for how
        to order the other three atoms.
        ref: Symmetrization of the AMBER and CHARMM Force Fields, J. Comput. Chem.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multipole and Gaussian
        Multipole Models
        """
        improper_id = 0
        for atom in self.atom:
            atom_symbol, neighbors = atom.GetSymbol(), atom.GetNeighbors()
            if atom_symbol not in csymbols or len(neighbors) != 3:
                continue
            if atom.GetSymbol() == symbols.NITROGEN and atom.GetHybridization(
            ) == Chem.rdchem.HybridizationType.SP3:
                continue
            # Sp2 carbon for planar, Sp3 with one H (CHR1R2R3) for chirality,
            # Sp2 N in Amino Acid
            improper_id += 1
            neighbor_symbols = [x.GetSymbol() for x in neighbors]
            counted = self.countSymbols([atom_symbol] + neighbor_symbols)
            improper_type_id = self.symbol_impropers[counted][0]
            # FIXME: see docstring for current investigation. (NO ACTIONS TAKEN)
            #  1) LAMMPS recommends the first to be the center, while the prm
            #  and literature order the third as the center.
            #  2) In addition, since improper has one non-connected edge,
            #  are the two non-edge atom selections important?
            #  3) Moreover, do we have to delete over constrained angle? If so,
            #  how about the one facing the non-connected edge?
            # My recommendation (not current implementation):
            # first plane: center + the two most heavy atom
            # second plane: the three non-center atoms
            # benefit: 1) O-C-O / O.O.R imposes symmetricity (RCOO)
            # 2) R-N-C / O.O.H exposes hydrogen out of plane vibration (RCNH)

            # My Implementation:
            # Use the center as the third according to "A New Force Field for
            # Molecular Mechanical Simulation of Nucleic Acids and Proteins"
            # No special treatment to the order of other atoms.

            # My Reasoning: first or third functions the same for planar
            # scenario as both 0 deg and 180 deg implies in plane. However,
            # center as first or third defines different planes, leading to
            # eiter ~45 deg or 120 deg as the equilibrium improper angle.
            # 120 deg sounds more plausible and thus the third is chosen to be
            # the center.
            atoms = [neighbors[0], neighbors[1], atom, neighbors[2]]
            self.impropers[improper_id] = (improper_type_id, ) + tuple(
                x.GetIntProp(self.ATOM_ID) for x in atoms)

    def printImpropers(self):
        """
        Print all the possible improper angles in the force field file.
        """
        for symb, improper_ids in self.symbol_impropers.items():
            print(f"{symb} {self.ff.impropers[improper_ids[0]]}")
            impropers = [self.ff.impropers[x] for x in improper_ids]
            for improper in impropers:
                ids = [improper.id1, improper.id2, improper.id3, improper.id4]
                print(f"{[self.ff.atoms[x].description for x in ids]}")

    def removeAngles(self):
        """
        One improper adds one restraint and thus one angle is removed.

        e.g. NH3 if all three H-N-H angles are defined, you cannot control out
        of plane mode.

        Two conditions are satisfied:
            1) the number of internal geometry variables is Nv= 3N_atom – 6
            2) each variable can be perturbed independently of the other variables
        For the case of ammonia, 3 bond lengths N-H1, N-H2, N-H3, the two bond
        angles θ1 = H1-N-H2 and θ2 = H1-N-H3, and the ω = H2-H1-N-H3
        ref: Atomic Forces for Geometry-Dependent Point Multipole and Gaussian
        Multipole Models
        """

        for idx, (itype, id1, id2, id3, id4) in self.impropers.items():
            angle_atom_ids = tuple([id1, id3, id4])
            if angle_atom_ids not in self.rvrs_angles:
                angle_atom_ids = angle_atom_ids[::-1]
            self.angles.pop(self.rvrs_angles[angle_atom_ids])

    def removeUnused(self):
        """
        Remove used force field information so that the data file is minimal.
        """
        if not self.concise:
            return

        atypes = sorted(set(x.GetIntProp(self.TYPE_ID) for x in self.atom))
        self.atm_types = {y: x for x, y in enumerate(atypes, start=1)}
        btypes = sorted(set(x[0] for x in self.bonds.values()))
        self.bnd_types = {y: x for x, y in enumerate(btypes, start=1)}
        antypes = sorted(set(x[0] for x in self.angles.values()))
        self.ang_types = {y: x for x, y in enumerate(antypes, start=1)}
        dtps = sorted(set(x[0] for x in self.dihedrals.values()))
        self.dihe_types = {y: x for x, y in enumerate(dtps, start=1)}
        itps = sorted(set(x[0] for x in self.impropers.values()))
        self.impr_types = {y: x for x, y in enumerate(itps, start=1)}

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom, bond,
        angle etc.
        """
        if self.mols is None:
            raise ValueError(f"Mols are not set.")

        self.data_fh.write(f"{self.LAMMPS_DESCRIPTION}\n\n")
        atom_nums = [len(x.GetAtoms()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(atom_nums)} {self.ATOMS}\n")
        self.data_fh.write(f"{len(self.bonds)} {self.BONDS}\n")
        self.data_fh.write(f"{len(self.angles)} {self.ANGLES}\n")
        self.data_fh.write(f"{len(self.dihedrals)} {self.DIHEDRALS}\n")
        self.data_fh.write(f"{len(self.impropers)} {self.IMPROPERS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        atom_num = len(self.atm_types) if self.concise else len(self.ff.atoms)
        self.data_fh.write(f"{atom_num} {self.ATOM_TYPES}\n")
        bond_num = len(self.bnd_types) if self.concise else len(self.ff.bonds)
        self.data_fh.write(f"{bond_num} {self.BOND_TYPES}\n")
        ang_num = len(self.ang_types) if self.concise else len(self.ff.angles)
        self.data_fh.write(f"{ang_num} {self.ANGLE_TYPES}\n")
        dnum = len(self.dihe_types) if self.concise else len(self.ff.dihedrals)
        self.data_fh.write(f"{dnum} {self.DIHEDRAL_TYPES}\n")
        inum = len(self.impr_types) if self.concise else len(self.ff.impropers)
        self.data_fh.write(f"{inum} {self.IMPROPER_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        """
        Write box information.

        :param min_box list: minimum box size
        :param buffer list: buffer in three dimensions
        """

        xyzs = np.concatenate(
            [x.GetConformer(0).GetPositions() for x in self.mols.values()])
        centroid = xyzs.mean(axis=0)
        box_hf = self.getHalfBox(xyzs, min_box=min_box, buffer=buffer)
        for dim in range(3):
            self.data_fh.write(
                f"{centroid[dim]-box_hf[dim]:.2f} {centroid[dim]+box_hf[dim]:.2f} "
                f"{self.LO_HI[dim]}\n")
        self.data_fh.write("\n")

    def getHalfBox(self, xyzs, min_box=None, buffer=None):
        """
        Get the half box size based on interaction minimum, buffer, and structure
        span.

        :return list of three floats: the xyz box limits.
        """

        if self.box is not None:
            return [(x - y) * 0.5
                    for x, y in zip(self.box[1::2], self.box[::2])]

        if min_box is None:
            # PBC should be 2x larger than the cutoff, otherwise one particle
            # can interact with another particle within its cutoff twice: within
            # the box and across the PBC.
            cut_x2 = min([self.lj_cut, self.coul_cut]) * 2
            min_box = (cut_x2, cut_x2, cut_x2,)  # yapf: disable
        if buffer is None:
            buffer = self.BUFFER  # yapf: disable
            box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
            box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]

        if len(self.mols) != 1:
            return box_hf
        # All-trans single molecule with internal tension runs into clashes
        # across PBCs and thus larger box is used.

        return [x * 1.2 for x in box_hf]

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.data_fh.write(f"{self.MASSES}\n\n")
        for atom_id, atom in self.ff.atoms.items():
            if self.concise and atom_id not in self.atm_types:
                continue
            atm_id = self.atm_types[atom_id] if self.concise else atom_id
            dscrptn = f"{atom.description} {atom.symbol} {atom_id}" if self.concise else atom.description
            self.data_fh.write(f"{atm_id} {atom.mass} # {dscrptn}\n")
        self.data_fh.write(f"\n")

    def writePairCoeffs(self):
        """
        Write pair coefficients.
        """
        self.data_fh.write(f"{self.PAIR_COEFFS}\n\n")
        for atom in self.ff.atoms.values():
            if self.concise and atom.id not in self.atm_types:
                continue
            vdw = self.ff.vdws[atom.id]
            atom_id = self.atm_types[atom.id] if self.concise else atom.id
            self.data_fh.write(f"{atom_id} {vdw.ene:.4f} {vdw.dist:.4f}\n")
        self.data_fh.write("\n")

    def writeBondCoeffs(self):
        """
        Write bond coefficients.
        """

        if not self.bnd_types:
            return

        self.data_fh.write(f"{self.BOND_COEFFS}\n\n")
        for bond in self.ff.bonds.values():
            if self.concise and bond.id not in self.bnd_types:
                continue
            bond_id = self.bnd_types[bond.id] if self.concise else bond.id
            self.data_fh.write(f"{bond_id}  {bond.ene} {bond.dist}\n")
        self.data_fh.write("\n")

    def writeAngleCoeffs(self):
        """
        Write angle coefficients.
        """
        if not self.ang_types:
            return

        self.data_fh.write(f"{self.ANGLE_COEFFS}\n\n")
        for angle in self.ff.angles.values():
            if self.concise and angle.id not in self.ang_types:
                continue
            angle_id = self.ang_types[angle.id] if self.concise else angle.id
            self.data_fh.write(f"{angle_id} {angle.ene} {angle.angle}\n")
        self.data_fh.write("\n")

    def writeDihedralCoeffs(self):
        """
        Write dihedral coefficients.
        """
        if not self.dihe_types:
            return

        self.data_fh.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        for dihe in self.ff.dihedrals.values():
            if self.concise and dihe.id not in self.dihe_types:
                continue
            dihedral_id = self.dihe_types[dihe.id] if self.concise else dihe.id
            params = [0., 0., 0., 0.]
            # LAMMPS: K1, K2, K3, K4 in 0.5*K1[1+cos(x)] + 0.5*K2[1-cos(2x)]...
            # OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            for ene_ang_n in dihe.constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if (ene_ang_n.angle == 180.) ^ (not ene_ang_n.n_parm % 2):
                    params[ene_ang_n.n_parm] *= -1
            self.data_fh.write(
                f"{dihedral_id}  {' '.join(map(str, params))}\n")
        self.data_fh.write("\n")

    def writeImproperCoeffs(self):
        """
        Write improper coefficients.
        """
        if not self.impr_types:
            return

        self.data_fh.write(f"{self.IMPROPER_COEFFS}\n\n")
        for impr in self.ff.impropers.values():
            if self.concise and impr.id not in self.impr_types:
                continue
            improper_id = self.impr_types[impr.id] if self.concise else impr.id
            # LAMMPS: K in K[1+d*cos(nx)] vs OPLS: [1 + cos(nx-gama)]
            # due to cos (θ - 180°) = cos (180° - θ) = - cos θ
            sign = 1 if impr.angle == 0. else -1
            self.data_fh.write(
                f"{improper_id} {impr.ene} {sign} {impr.n_parm}\n")
        self.data_fh.write("\n")

    def writeAtoms(self):
        """
        Write atom coefficients.

        :param comments bool: If True, additional descriptions including element
            sysmbol are written after each atom line
        """
        def get_neigh_charge(atom):
            try:
                return atom.GetDoubleProp(self.NEIGHBOR_CHARGE)
            except KeyError:
                return 0

        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        for mol_id, mol in self.mols.items():
            data = np.zeros((mol.GetNumAtoms(), 7))
            conformer = mol.GetConformer()
            data[:, 0] = [x.GetIntProp(self.ATOM_ID) for x in mol.GetAtoms()]
            data[:, 1] = mol_id
            type_ids = [x.GetIntProp(self.TYPE_ID) for x in mol.GetAtoms()]
            data[:, 2] = [
                self.atm_types[x] if self.concise else x for x in type_ids
            ]
            charges = [get_neigh_charge(x) for x in mol.GetAtoms()]
            data[:, 3] = [
                x + self.ff.charges[y] for x, y in zip(charges, type_ids)
            ]
            data[:, 4:] = conformer.GetPositions()
            np.savetxt(self.data_fh, data, fmt='%i %i %i %.4f %.3f %.3f %.3f')
        self.data_fh.write(f"\n")

    def writeBonds(self):
        """
        Write bond coefficients.
        """

        if not self.bonds:
            return

        self.data_fh.write(f"{self.BONDS.capitalize()}\n\n")
        for bond_id, (bond_type, id1, id2) in self.bonds.items():
            bond_type = self.bnd_types[bond_type] if self.concise else bond_type
            self.data_fh.write(f"{bond_id} {bond_type} {id1} {id2}\n")
        self.data_fh.write(f"\n")

    def writeAngles(self):
        """
        Write angle coefficients.
        """
        if not self.angles:
            return
        self.data_fh.write(f"{self.ANGLES.capitalize()}\n\n")
        # Some angles may be filtered out by improper
        for angle_id, value in enumerate(self.angles.items(), start=1):
            _, (type_id, id1, id2, id3) = value
            angle_type = self.ang_types[type_id] if self.concise else type_id
            self.data_fh.write(f"{angle_id} {angle_type} {id1} {id2} {id3}\n")
        self.data_fh.write(f"\n")

    def writeDihedrals(self):
        """
        Write dihedral coefficients.
        """
        if not self.dihedrals:
            return

        self.data_fh.write(f"{self.DIHEDRALS.capitalize()}\n\n")
        for dihe_id, (type_id, id1, id2, id3, id4) in self.dihedrals.items():
            type_id = self.dihe_types[type_id] if self.concise else type_id
            self.data_fh.write(
                f"{dihe_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_fh.write(f"\n")

    def writeImpropers(self):
        """
        Write improper coefficients.
        """
        if not self.impropers:
            return

        self.data_fh.write(f"{self.IMPROPERS.capitalize()}\n\n")
        for improper_id, (type_id, id1, id2, id3,
                          id4) in self.impropers.items():
            type_id = self.impr_types[type_id] if self.concise else type_id
            self.data_fh.write(
                f"{improper_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_fh.write(f"\n")


class DataFileReader(LammpsData):
    """
    LAMMPS Data file reader
    """

    SCALE = 0.5

    def __init__(self, data_file=None, min_dist=2., contents=None):
        """
        :param data_file str: data file with path
        :param min_dist: the minimum distance as clash (some h-bond has zero vdw
            params)
        :param contents `bytes`: parse the contents if data_file not provided.
        """
        self.data_file = data_file
        self.min_dist = min_dist
        self.contents = contents
        self.lines = None
        self.masses = {}
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.vdws = {}
        self.radii = {}
        self.mols = {}
        self.excluded = collections.defaultdict(set)

    def run(self):
        """
        Main method to read and parse the data file.
        """
        self.read()
        self.indexLines()
        self.setDescription()
        self.setMasses()
        self.setPairCoeffs()
        self.setAtoms()
        self.setBonds()
        self.setAngles()
        self.setDihedrals()
        self.setImpropers()
        self.setMols()

    def setMinimumDist(self):
        for id in self.vdws.keys():
            if self.vdws[id].dist < self.min_dist:
                self.vdws[id].dist = self.min_dist

    def read(self):
        """
        Read the data file and index lines by section marker.
        """
        if self.data_file:
            with open(self.data_file, 'r') as df_fh:
                self.lines = df_fh.readlines()
        else:
            content_type, content_string = self.contents.split(',')
            decoded = base64.b64decode(content_string)
            self.lines = decoded.decode("utf-8").splitlines()

    def indexLines(self):
        self.mk_idxes = {
            x: i
            for i, l in enumerate(self.lines) for x in self.MARKERS
            if l.startswith(x)
        }

    def setDescription(self):
        """
        Parse the description section for topo counts, type counts, and box size
        """
        dsp_eidx = min(self.mk_idxes.values())
        # {'atoms': 1620, 'bonds': 1593, 'angles': 1566, 'dihedrals': 2511}
        self.struct_dsp = {
            y: int(self.lines[x].split(y)[0])
            for x in range(dsp_eidx) for y in self.STRUCT_DSP
            if y in self.lines[x]
        }
        # {'atom types': 7, 'bond types': 6, 'angle types': 5}
        self.dype_dsp = {
            y: int(self.lines[x].split(y)[0])
            for x in range(dsp_eidx) for y in self.TYPE_DSP
            if y in self.lines[x]
        }
        # {'xlo xhi': [-7.12, 35.44], 'ylo yhi': [-7.53, 34.26], 'zlo zhi': ..}
        self.box_dsp = {
            y: [float(z) for z in self.lines[x].split(y)[0].split()]
            for x in range(dsp_eidx) for y in self.BOX_DSP
            if y in self.lines[x]
        }

    def setMasses(self):
        """
        Parse the mass section for masses and elements.
        """
        sidx = self.mk_idxes[self.MASSES] + 2
        for id, lid in enumerate(
                range(sidx, sidx + self.dype_dsp[self.ATOM_TYPES]), 1):
            splitted = self.lines[lid].split()
            id, mass, ele = splitted[0], splitted[1], splitted[-2]
            self.masses[int(id)] = types.SimpleNamespace(id=int(id),
                                                         mass=float(mass),
                                                         ele=ele)

    def setAtoms(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        sidx = self.mk_idxes[self.ATOMS_CAP] + 2
        for lid in range(sidx, sidx + self.struct_dsp[self.ATOMS]):
            id, mol_id, type_id, charge, x, y, z = self.lines[lid].split()[:7]
            self.atoms[int(id)] = types.SimpleNamespace(
                id=int(id),
                mol_id=int(mol_id),
                type_id=int(type_id),
                xyz=(float(x), float(y), float(z)),
                ele=self.masses[int(type_id)].ele)

    @property
    def atom(self):
        """
        Handy way to get all atoms.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return (x for x in self.atoms.values())

    @property
    def atom_num(self):
        """
        Handy way to get all atoms.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return len(self.atoms)

    @property
    def molecule(self):
        """
        Handy way to get all molecules.

        :return list of list: each sublist contains one int as atom id
        """

        return super().molecule

    def setMols(self):
        """
        Group atoms into molecules by molecule ids.
        """
        mols = collections.defaultdict(list)
        for atom in self.atoms.values():
            mols[atom.mol_id].append(atom.id)
        self.mols = dict(mols)

    def setBonds(self):
        """
        Parse the atom section for atom id and molecule id.
        """
        if self.BONDS_CAP not in self.mk_idxes:
            return

        sidx = self.mk_idxes[self.BONDS_CAP] + 2
        for lid in range(sidx, sidx + self.struct_dsp[self.BONDS]):
            id, type_id, id1, id2 = self.lines[lid].split()
            self.bonds[int(id)] = types.SimpleNamespace(id=int(id),
                                                        type_id=int(type_id),
                                                        id1=int(id1),
                                                        id2=int(id2))

    def setAngles(self):
        """
        Parse the angle section for angle id and constructing atoms.
        """
        try:
            sidx = self.mk_idxes[self.ANGLES_CAP] + 2
        except KeyError:
            return
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.ANGLES]), 1):

            id, type_id, id1, id2, id3 = self.lines[lid].split()[:5]
            self.angles[int(id)] = types.SimpleNamespace(id=int(id),
                                                         type_id=int(type_id),
                                                         id1=int(id1),
                                                         id2=int(id2),
                                                         id3=int(id3))

    def setDihedrals(self):
        """
        Parse the dihedral section for dihedral id and constructing atoms.
        """
        try:
            sidx = self.mk_idxes[self.DIHEDRALS_CAP] + 2
        except KeyError:
            return
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.DIHEDRALS]), 1):
            id, type_id, id1, id2, id3, id4 = self.lines[lid].split()[:6]
            self.dihedrals[int(id)] = types.SimpleNamespace(
                id=int(id),
                type_id=int(type_id),
                id1=int(id1),
                id2=int(id2),
                id3=int(id3),
                id4=int(id4))

    def setImpropers(self):
        """
        Parse the improper section for dihedral id and constructing atoms.
        """
        try:
            sidx = self.mk_idxes[self.IMPROPERS_CAP] + 2
        except KeyError:
            return
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.IMPROPERS]), 1):
            id, type_id, id1, id2, id3, id4 = self.lines[lid].split()[:6]
            self.impropers[int(id)] = types.SimpleNamespace(
                id=int(id),
                type_id=int(type_id),
                id1=int(id1),
                id2=int(id2),
                id3=int(id3),
                id4=int(id4))

    def setClashParams(self, include14=False, scale=SCALE):
        """
        Set clash check related parameters including pair radii and exclusion.

        :param include14 bool: whether to include atom separated by 2 bonds for
            clash check.
        :param scale float: the scale param on vdw radius in clash check.
        """
        self.setClashExclusion(include14=not include14)
        self.setPairCoeffs()
        self.setVdwRadius(scale=scale)

    def setClashExclusion(self, include14=True):
        """
        Bonded atoms and atoms in angles are in the exclusion. If include14=True,
        the dihedral angles are in the exclusion as well.

        :param include14 bool: If True, 1-4 interaction in a dihedral angle count
            as exclusion.
        """
        pairs = set((x.id1, x.id2) for x in self.bonds.values())
        pairs = pairs.union((x.id1, x.id3) for x in self.angles.values())
        pairs = pairs.union([
            y for x in self.impropers.values()
            for y in itertools.combinations([x.id1, x.id2, x.id3, x.id4], 2)
        ])
        if include14:
            pairs = pairs.union(
                (x.id1, x.id4) for x in self.dihedrals.values())
        for id1, id2 in pairs:
            self.excluded[id1].add(id2)
            self.excluded[id2].add(id1)

    def setPairCoeffs(self):
        """
        Paser the pair coefficient section.
        """
        sidx = self.mk_idxes[self.PAIR_COEFFS] + 2
        for lid in range(sidx, sidx + self.dype_dsp[self.ATOM_TYPES]):
            id, ene, dist = self.lines[lid].split()
            self.vdws[int(id)] = types.SimpleNamespace(id=int(id),
                                                       dist=float(dist),
                                                       ene=float(ene))

    def setVdwRadius(self, mix=LammpsData.GEOMETRIC, scale=1.):
        """
        Set the vdw radius based on the mixing rule and vdw radii.

        :param mix str: the mixing rules, including GEOMETRIC, ARITHMETIC, and
            SIXTHPOWER
        :param scale float: scale the vdw radius by this factor

        NOTE: the scaled radii here are more like diameters (or distance)
            between two sites.
        """
        if mix == LammpsData.GEOMETRIC:
            # LammpsData.GEOMETRIC is optimized for speed and is supported
            radii = [0] + [
                self.vdws[x.type_id].dist for x in self.atoms.values()
            ]
            shape = len(self.atoms) + 1
            self.radii = np.full((shape, shape), radii, dtype='float16')
            self.radii[:, 0] = self.radii[0, :]
            self.radii *= self.radii.transpose()
            self.radii = np.sqrt(self.radii)
            self.radii *= pow(2, 1 / 6) * scale
            self.radii[self.radii < self.min_dist] = self.min_dist
            return

        radii = collections.defaultdict(dict)
        for id1, vdw1 in self.vdws.items():
            for id2, vdw2 in self.vdws.items():
                if mix == self.GEOMETRIC:
                    dist = pow(vdw1.dist * vdw2.dist, 0.5)
                elif mix == self.ARITHMETIC:
                    dist = (vdw1.dist + vdw2.dist) / 2
                elif mix == self.SIXTHPOWER:
                    dist = (pow(vdw1.dist, 6) + pow(vdw2.dist, 6)) / 2
                    dist = pow(dist, 1 / 6)
                dist *= pow(2, 1 / 6) * scale
                if dist < self.min_dist:
                    dist = self.min_dist
                radii[id1][id2] = round(dist, 4)

        self.radii = collections.defaultdict(dict)
        for atom1 in self.atoms.values():
            for atom2 in self.atoms.values():
                self.radii[atom1.id][atom2.id] = radii[atom1.type_id][
                    atom2.type_id]
        self.radii = dict(self.radii)
