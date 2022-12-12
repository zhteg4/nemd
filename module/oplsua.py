import sys
import types
import symbols
import chemparse
import itertools
import logutils
import fileutils
import constants
import collections
import environutils
import numpy as np
from rdkit import Chem
from collections import namedtuple

BOND_ATM_ID = 'bond_atm_id'
RES_NUM = 'res_num'
IMPLICIT_H = 'implicit_h'
TYPE_ID = 'type_id'
LARGE_NUM = constants.LARGE_NUM

ATOM_TYPE = namedtuple('ATOM_TYPE', [
    'id', 'formula', 'symbol', 'description', 'atomic_number', 'mass',
    'connectivity'
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
    :return 'OPLS_Parser': the parser with force field information
    """
    opls_parser = OPLS_Parser()
    opls_parser.read()
    return opls_parser


class OPLS_Typer:
    """
    Type the atoms and map SMILES fragments.
    """

    TYPE_ID = TYPE_ID
    RES_NUM = RES_NUM
    BOND_ATM_ID = BOND_ATM_ID
    IMPLICIT_H = IMPLICIT_H
    LARGE_NUM = constants.LARGE_NUM

    # yapf: disable
    SMILES = [UA(sml='C', mp=(81, ), hs=None, dsc='CH4 Methane'),
              UA(sml='CC', mp=(82, 82,), hs=None, dsc='Ethane'),
              UA(sml='CO', mp=(106, 104,), hs={104:105}, dsc='Ethane'),
              UA(sml='CCC', mp=(83, 86, 83,), hs=None, dsc='Propane'),
              UA(sml='CCCC', mp=(83, 86, 86, 83,), hs=None, dsc='n-Butane'),
              UA(sml='CC(C)C', mp=(84, 88, 84, 84, ), hs=None, dsc='Isobutane'),
              UA(sml='CC=CC', mp=(84, 89, 89, 84, ), hs=None, dsc='2-Butene'),
              UA(sml='CN(C)C=O', mp=(156, 148, 156, 153, 151,), hs=None, dsc='N,N-Dimethylformamide'),
              # "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
              UA(sml='O=CO', mp=(134, 133, 135), hs={135: 136}, dsc='Carboxylic Acid'),
              # "Methyl", "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
              UA(sml='CC(=O)O', mp=(137, 133, 134, 135), hs={135: 136}, dsc='Ethanoic acid')]
    # yapf: enable
    SMILES = list(reversed(SMILES))
    ATOM_TOTAL = {i: i for i in range(1, 214)}
    BOND_ATOM = ATOM_TOTAL.copy()
    # "O Peptide Amide" "COH (zeta) Tyr" "OH Tyr"  "H(O) Ser/Thr/Tyr"
    BOND_ATOM.update({134: 2, 133: 26, 135: 23, 136: 24, 153: 72, 148: 3})
    ANGLE_ATOM = ATOM_TOTAL.copy()
    ANGLE_ATOM.update({134: 2, 133: 17, 135: 76, 136: 24, 148: 3, 153: 72})
    DIHE_ATOM = ATOM_TOTAL.copy()
    DIHE_ATOM.update({134: 11, 133: 26, 135: 76, 136: 24, 148: 3, 153: 72})
    # C-OH (Tyr) is used as HO-C=O, which needs CH2-COOH map as alpha-COOH bond
    BOND_ATOMS = {(26, 86): [16, 17], (26, 88): [16, 17]}
    # yapf: disable
    DIHE_ATOMS = {(26,86,): (1,6,), (26,88,): (1,6,)}
    # yapf: enable
    DESCRIPTION_SMILES = {
        'CH4 Methane': 'C',
        'Ethane': 'CC',
        'n-Butane': 'CCCC',
        "Isobutane": 'CC(C)C',
        "CH2 (generic)": '*-C-*',
        "Benzene": 'C1=CC=CC=C1',
        "Phenol": '	OC1=CC=CC=C1',
        "2-Butene": 'CC=CC',
        'Li+ Lithium Ion': '[Li+]',
        'Na+ Sodium Ion': '[Na+]',
        'K+ Potassium Ion': '[K+]',
        'Rb+ Rubidium Ion': '[Rb+]',
        'Cs+ Cesium Ion': '[Cs+]',
        'Mg+2 Magnesium Ion': '[Mg+]',
        'Ca+2 Calcium Ion': '[Ca+]',
        'Sr+2 Strontium Ion': '[Sr+]',
        'Ba+2 Barium Ion': '[Ba+]',
        'F- Fluoride Ion': '[F-]',
        'Cl- Chloride Ion': '[Cl-]',
        'Br- Bromide Ion': '[Br-]',
        'Helium Atom': '[He]',
        'Neon Atom': '[Ne]',
        'Argon Atom': '[Ar]',
        'Krypton Atom': '[Kr]',
        'Xenon Atom': '[Xe]',
        'Peptide Amide': '*-NC(-*)=O',
        'Acetone': ' CC(C)=O',
        'Carboxylic Acid': 'OC(-*)=O',
        'Methyl Acetate': 'COC(C)=O',
        'Alcohol Hydroxyl': '*-O',
        'Carboxylate': '[O-]C(-*)=O',
        'Hydrogen Sulfide': 'S',
        "SH Alkyl Sulfide": '*-CS',
        'Methyl Sulfide': '*SC',
        'Ethyl Sulfide': '*SCC'
    }

    def __init__(self, mol):
        self.mol = mol

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
        log_debug(f"{len(marked_atom_ids)}, {self.mol.GetNumAtoms()}")
        log_debug(f"{res_num - 1} residues found.")
        [log_debug(f'{x}: {y}') for x, y in marked_smiles.items()]

    def filterMatch(self, match, frag):
        """
        Filter substruct matches based on connectivity.

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
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == symbols.HYDROGEN:
                    type_id = sml.hs[type_id]
                    self.markAtom(neighbor, type_id, res_num)
                    marked.append(neighbor.GetIdx())
                    log_debug(
                        f"{neighbor.GetSymbol()}{neighbor.GetDegree()} {neighbor.GetIdx()} {type_id}"
                    )
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


class OPLS_Parser(OPLS_Typer):
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

    def __init__(self):
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
            h_count = prsd.pop(symbols.HYDROGEN, 0)
            symbol = [x for x in prsd.keys()][0] if prsd else symbols.HYDROGEN
            self.atoms[int(id)] = ATOM_TYPE(id=int(id),
                                            formula=formula,
                                            symbol=symbol,
                                            description=comment,
                                            atomic_number=int(atomic_number),
                                            mass=float(mass),
                                            connectivity=int(cnnct) + h_count)

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
        :return: 
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

        atom_types = [x.GetIntProp(self.BOND_ATM_ID) for x in bonded_atoms]
        try:
            atom_types = OPLS_Parser.BOND_ATOMS[tuple(sorted(atom_types))]
        except KeyError:
            # C-OH (Tyr) is used as HO-C=O, needing CH2-COOH map as alpha-COOH bond
            pass
        # Exact match between two atom type ids
        matches = [
            x for x in self.bonds.values() if [x.id1, x.id2] == atom_types
        ]
        if matches:
            return matches

        log_debug(
            f"No exact params for bond between atom type {atom_types[0]} and {atom_types[1]}."
        )
        bond_score, type_set = {}, set(atom_types)
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
            scnnt = self.atoms[atom_id].connectivity == self.getAtomConnt(atom)
            bond_score[bond] = [ssymbol, scnnt]

        matches = [x for x, y_z in bond_score.items() if all(y_z)]
        if not matches:
            matches = [x for x, (y, z) in bond_score.items() if y]
        if not matches:
            raise ValueError(
                f"No params for bond between atom type {atom_types[0]} and {atom_types[1]}."
            )
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

        :param atoms:
        :return:
        """


        end_ids = [self.ANGLE_ATOM[x.GetIntProp(self.TYPE_ID)] for x in atoms[::2]]
        if end_ids[0] > end_ids[1]:
            atoms = list(reversed(atoms))
        type_ids = [x.GetIntProp(self.TYPE_ID) for x in atoms]
        type_ids = [OPLS_Parser.ANGLE_ATOM[x] for x in type_ids]
        matches = [
            x for x in self.angles.values()
            if type_ids == [x.id1, x.id2, x.id3]
        ]

        if matches:
            return matches
        log_debug(
            f"No exact params for angle between atom {', '.join(map(str, type_ids))}."
        )
        partial_matches = [
            x for x in self.angles.values() if x.id2 == type_ids[1]
        ]
        if not partial_matches:
            import pdb;pdb.set_trace()
            raise ValueError(
                f"No params for angle (middle atom type {type_ids[1]}).")
        matches = self.getMatchesFromEnds(atoms, partial_matches)
        if not matches:
            raise ValueError(
                f"No params for angle between atom {', '.join(map(str, type_ids))}."
            )
        self.debugPrintReplacement(atoms, matches)
        return matches

    def debugPrintReplacement(self, atoms, matches):
        smbl_cnnts = [f'{x.GetSymbol()}{self.getAtomConnt(x)}' for x in atoms]
        ids = [
            getattr(matches[0], x, '') for x in ['id1', 'id2', 'id3', 'id4']
        ]
        ids = [x for x in ids if x]
        id_smbl_cnnts = [
            f'{self.atoms[x].symbol}{self.atoms[x].connectivity}'
            for x in ids
        ]
        log_debug(
            f"{'~'.join(smbl_cnnts)} "
            f"{'~'.join(map(str, [x.GetIntProp(self.TYPE_ID) for x in atoms]))} "
            f"replaced by {'~'.join(map(str, id_smbl_cnnts))} {'~'.join(map(str, ids))}"
        )

    def getMatchesFromEnds(self, atoms, partial_matches, rough=False):
        o_symbols = set((
            x.GetSymbol(),
            self.getAtomConnt(x),
        ) for x in [atoms[0], atoms[-1]])
        ff_atom_ids = [
            [x, x.id1, x.id4] if hasattr(x, 'id4') else [x, x.id1, x.id3]
            for x in partial_matches
        ]
        ff_symbols = {
            x[0]: set([(
                self.atoms[y].symbol,
                self.atoms[y].connectivity,
            ) for y in x[1:]])
            for x in ff_atom_ids
        }
        matches = [x for x, y in ff_symbols.items() if y == o_symbols]
        if not matches:
            o_symbols_partial = set(x[0] for x in o_symbols)
            matches = [
                x for x, y in ff_symbols.items()
                if set(z[0] for z in y) == o_symbols_partial
            ]
        return matches

class LammpsIn(fileutils.LammpsInput):
    """
    Class to write out LAMMPS in script.
    """

    IN_EXT = '.in'

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
        :param timestep float: timestep size for subsequent molecular dynamics simulations
        """
        self.jobname = jobname
        self.lj_cut = lj_cut
        self.coul_cut = coul_cut
        self.timestep = timestep
        self.concise = concise
        self.lammps_in = self.jobname + self.IN_EXT
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
            self.in_fh.write(f"{self.KSPACE_STYLE} 0.0001\n")
        self.in_fh.write(f"log log.{self.jobname}\n")

    def hasCharge(self):
        """
        Whether any atom has charge.
        """
        charges = [
            self.ff.charges[y.GetIntProp(self.TYPE_ID)]
            for x in self.mols.values() for y in x.GetAtoms()
        ]
        return any(charges)

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
            atoms = self.ff.atoms.values()
            if self.concise:
                atoms = [x for x in atoms if x.id in self.used_atom_types]
            smbs = ' '.join(map(str, [x.symbol for x in atoms]))
            self.in_fh.write(f"dump_modify 1 element {smbs}\n")
        self.in_fh.write(f"{self.MIN_STYLE} {min_style}\n")
        self.in_fh.write("minimize 1.0e-6 1.0e-8 1000 100000\n")

    def writeTimestep(self):
        """
        Write commands related to timestep.
        """
        self.in_fh.write(f'timestep {self.timestep}\n')
        self.in_fh.write('thermo_modify flush yes\n')
        self.in_fh.write('thermo 1000\n')

    def writeRun(self,
                 start_temp=10.,
                 target_temp=None,
                 tdamp=100,
                 pdamp=1000):
        """
        Write command to further equilibrate the system.

        :param start_temp float: the starting temperature for relaxation
        :param target_temp float: the target temperature at the end of the relaxation
        :param tdamp float: damping factor to control temperature
        :param pdamp float: damping factor to control pressure
        """

        self.in_fh.write(f"velocity all create {start_temp} 482748\n")
        if len(self.mols) == 1 and self.mols[1].GetNumAtoms() < 10:
            # NVT on single molecule gives nan coords (guess due to translation)
            self.in_fh.write("fix 1 all nve\n")
            self.in_fh.write("run 2000\n")
            return

        self.in_fh.write(
            f"fix 1 all nvt temp {start_temp} {start_temp} {self.timestep * tdamp}\n"
        )
        self.in_fh.write("run 2000\n")

        if len(self.mols) == 1:
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


class LammpsWriter(LammpsIn):
    """
    Class to write out LAMMPS data file.
    """

    DATA_EXT = '.data'
    LAMMPS_DESCRIPTION = 'LAMMPS Description'

    TYPE_ID = OPLS_Parser.TYPE_ID
    ATOM_ID = 'atom_id'
    RES_NUM = RES_NUM
    NEIGHBOR_CHARGE = 'neighbor_charge'
    BOND_ATM_ID = OPLS_Parser.BOND_ATM_ID
    IMPLICIT_H = OPLS_Parser.IMPLICIT_H

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

    def __init__(self, mols, ff, jobname, concise=True, *arg, **kwarg):
        """
        :param ff 'oplsua.OPLS_Parser':
        :param jobname str: jobname based on which out filenames are defined
        :param mols dict: keys are the molecule ids, and values are 'rdkit.Chem.rdchem.Mol'
        :param lj_cut float: cut off distance for Lennard-Jones potential
        :param coul_cut float: cut off distance for Coulombic pairwise interaction
        :param timestep float: timestep size for subsequent molecular dynamics simulations
        :param concise bool: If False, all the atoms in the force field file shows
            up in the force field section of the data file. If True, only the present
            ones are writen into the data file.
        """
        super().__init__(jobname, *arg, **kwarg)
        self.ff = ff
        self.mols = mols
        self.jobname = jobname
        self.concise = concise
        self.lammps_data = self.jobname + self.DATA_EXT
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.symbol_impropers = {}
        self.used_atom_types = []
        self.used_bond_types = []
        self.used_angle_types = []
        self.used_dihedral_types = []
        self.used_improper_types = []
        self.data_fh = None

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
            self.AnglesByImpropers()
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

        for atom_id, atom in enumerate(self.atom, start=1):
            atom.SetIntProp(self.ATOM_ID, atom_id)

    @property
    def atom(self):
        """
        Handy way to get all atoms.

        :return generator of 'rdkit.Chem.rdchem.Atom': all atom in all molecules
        """

        return (atom for mol in self.mols.values() for atom in mol.GetAtoms())

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
        bonds = [bond for mol in self.mols.values() for bond in mol.GetBonds()]
        for bond_id, bond in enumerate(bonds, start=1):
            bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
            # BOND_ATM_ID defines bonding parameters marked during atom typing
            bonded_atoms = sorted(bonded_atoms,
                                  key=lambda x: x.GetIntProp(self.BOND_ATM_ID))
            matches = self.ff.getMatchedBonds(bonded_atoms)
            bond = matches[0]
            self.bonds[bond_id] = (
                bond.id,
                bonded_atoms[0].GetIntProp(self.ATOM_ID),
                bonded_atoms[1].GetIntProp(self.ATOM_ID),
            )

    def adjustBondLength(self, adjust_bond_legnth=True):
        """
        Adjust bond length according to the force field parameters.

        :param adjust_bond_legnth bool: adjust bond length if True.
        """
        if not adjust_bond_legnth:
            return

        for mol_id, mol in self.mols.items():
            conformer = mol.GetConformer()
            for bond in mol.GetBonds():
                bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
                ids = set([x.GetIntProp(self.ATOM_ID) for x in bonded_atoms])
                mbond_type = [
                    x for x, y, z in self.bonds.values()
                    if len(ids.intersection([y, z])) == 2
                ][0]
                dist = self.ff.bonds[mbond_type].dist
                Chem.rdMolTransforms.SetBondLength(
                    conformer, *[x.GetIdx() for x in bonded_atoms], dist)

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

        all_angle_atoms = (y for x in self.atom for y in self.ff.getAngleAtoms(x))
        for angle_id, angle_atoms in enumerate(all_angle_atoms, start=1):
            angle = self.ff.getMatchedAngles(angle_atoms)[0]
            atom_ids = tuple(x.GetIntProp(self.ATOM_ID) for x in angle_atoms)
            self.angles[angle_id] = (angle.id, ) + atom_ids

    def removeUnused(self):
        """
        Remove used force field information so that the data file is minimal.
        """
        if not self.concise:
            return

        self.used_atom_types = [0] + sorted(
            set(
                y.GetIntProp(self.TYPE_ID) for x in self.mols.values()
                for y in x.GetAtoms()))
        self.used_bond_types = [0] + sorted(
            set(x[0] for x in self.bonds.values()))
        self.used_angle_types = [0] + sorted(
            set(x[0] for x in self.angles.values()))
        self.used_dihedral_types = [0] + sorted(
            set(x[0] for x in self.dihedrals.values()))
        self.used_improper_types = [0] + sorted(
            set(x[0] for x in self.impropers.values()))

    def writeDescription(self):
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
        atom_num = len(self.used_atom_types) - 1 if self.concise else len(
            self.ff.atoms)
        self.data_fh.write(f"{atom_num} {self.ATOM_TYPES}\n")
        bond_num = len(self.used_bond_types) - 1 if self.concise else len(
            self.ff.bonds)
        self.data_fh.write(f"{bond_num} {self.BOND_TYPES}\n")
        angle_num = len(self.used_angle_types) - 1 if self.concise else len(
            self.ff.angles)
        self.data_fh.write(f"{angle_num} {self.ANGLE_TYPES}\n")
        dihedral_num = len(
            self.used_dihedral_types) - 1 if self.concise else len(
                self.ff.dihedrals)
        self.data_fh.write(f"{dihedral_num} {self.DIHEDRAL_TYPES}\n")
        improper_num = len(
            self.used_improper_types) - 1 if self.concise else len(
                self.ff.impropers)
        self.data_fh.write(f"{improper_num} {self.IMPROPER_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        if min_box is None:
            min_box = (40., 40., 40.,) # yapf: disable
        if buffer is None:
            buffer = self.BUFFER # yapf: disable
        xyzs = np.concatenate(
            [x.GetConformer(0).GetPositions() for x in self.mols.values()])
        box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
        box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]
        if len(self.mols) == 1:
            box_hf = [x * 1.2 for x in box_hf]
        centroid = xyzs.mean(axis=0)
        for dim in range(3):
            self.data_fh.write(
                f"{centroid[dim]-box_hf[dim]:.2f} {centroid[dim]+box_hf[dim]:.2f} {self.LO_HI[dim]}\n"
            )
        self.data_fh.write("\n")

    def writeMasses(self):
        self.data_fh.write(f"{self.MASSES}\n\n")
        for atom_id, atom in self.ff.atoms.items():
            if self.concise and atom_id not in self.used_atom_types:
                continue
            atm_id = self.used_atom_types.index(
                atom_id) if self.concise else atom_id
            dscrptn = f"{atom.description} {atom_id}" if self.concise else atom.description
            self.data_fh.write(f"{atm_id} {atom.mass} # {dscrptn}\n")
        self.data_fh.write(f"\n")

    def writePairCoeffs(self):
        self.data_fh.write(f"{self.PAIR_COEFFS}\n\n")
        for atom in self.ff.atoms.values():
            if self.concise and atom.id not in self.used_atom_types:
                continue
            vdw = self.ff.vdws[atom.id]
            atom_id = self.used_atom_types.index(
                atom.id) if self.concise else atom.id
            self.data_fh.write(f"{atom_id} {vdw.ene} {vdw.dist}\n")
        self.data_fh.write("\n")

    def writeBondCoeffs(self):
        if len(self.used_bond_types) < 2 and self.concise:
            return

        self.data_fh.write(f"{self.BOND_COEFFS}\n\n")
        for bond in self.ff.bonds.values():
            if self.concise and bond.id not in self.used_bond_types:
                continue
            bond_id = self.used_bond_types.index(
                bond.id) if self.concise else bond.id
            self.data_fh.write(f"{bond_id}  {bond.ene} {bond.dist}\n")
        self.data_fh.write("\n")

    def writeAngleCoeffs(self):
        if len(self.used_angle_types) < 2 and self.concise:
            return

        self.data_fh.write(f"{self.ANGLE_COEFFS}\n\n")
        for angle in self.ff.angles.values():
            if self.concise and angle.id not in self.used_angle_types:
                continue
            angle_id = self.used_angle_types.index(
                angle.id) if self.concise else angle.id
            self.data_fh.write(f"{angle_id} {angle.ene} {angle.angle}\n")
        self.data_fh.write("\n")

    def writeDihedralCoeffs(self):
        if len(self.used_dihedral_types) < 2 and self.concise:
            return

        self.data_fh.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        for dihedral in self.ff.dihedrals.values():
            if self.concise and dihedral.id not in self.used_dihedral_types:
                continue
            dihedral_id = self.used_dihedral_types.index(
                dihedral.id) if self.concise else dihedral.id
            params = [0., 0., 0., 0.]
            for ene_ang_n in dihedral.constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if (ene_ang_n.angle == 180.) ^ (ene_ang_n.n_parm in (
                        2,
                        4,
                )):
                    params[ene_ang_n.n_parm] *= -1
            self.data_fh.write(
                f"{dihedral_id}  {' '.join(map(str, params))}\n")
        self.data_fh.write("\n")

    def writeImproperCoeffs(self):
        if len(self.used_dihedral_types) < 2 and self.concise:
            return

        self.data_fh.write(f"{self.IMPROPER_COEFFS}\n\n")
        for improper in self.ff.impropers.values():
            if self.concise and improper.id not in self.used_improper_types:
                continue
            improper_id = self.used_improper_types.index(
                improper.id) if self.concise else improper.id
            sign = 1 if improper.angle == 0. else -1
            self.data_fh.write(
                f"{improper_id} {improper.ene} {sign} {improper.n_parm}\n")
        self.data_fh.write("\n")

    def writeAtoms(self):
        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        for mol_id, mol in self.mols.items():
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                atom_id = atom.GetIntProp(self.ATOM_ID)
                type_id = atom.GetIntProp(self.TYPE_ID)
                xyz = conformer.GetAtomPosition(atom.GetIdx())
                xyz = ' '.join(map(lambda x: f'{x:.3f}', xyz))
                try:
                    ncharge = atom.GetDoubleProp(self.NEIGHBOR_CHARGE)
                except KeyError:
                    ncharge = 0
                charge = self.ff.charges[type_id] + ncharge
                dsrptn = self.ff.atoms[type_id].description
                symbol = self.ff.atoms[type_id].symbol
                type_id = self.used_atom_types.index(
                    type_id) if self.concise else type_id
                self.data_fh.write(
                    f"{atom_id} {mol_id} {type_id} {charge:.4f} {xyz} # {dsrptn} {symbol}\n"
                )
        self.data_fh.write(f"\n")

    def debugPrintReplacement(self, atoms, matches):
        smbl_cnnts = [f'{x.GetSymbol()}{self.getAtomConnt(x)}' for x in atoms]
        ids = [
            getattr(matches[0], x, '') for x in ['id1', 'id2', 'id3', 'id4']
        ]
        ids = [x for x in ids if x]
        id_smbl_cnnts = [
            f'{self.ff.atoms[x].symbol}{self.ff.atoms[x].connectivity}'
            for x in ids
        ]
        log_debug(
            f"{'~'.join(smbl_cnnts)} "
            f"{'~'.join(map(str, [x.GetIntProp(self.TYPE_ID) for x in atoms]))} "
            f"replaced by {'~'.join(map(str, id_smbl_cnnts))} {'~'.join(map(str, ids))}"
        )

    def getAtomConnt(self, atom):
        implicit_h_num = atom.GetIntProp(self.IMPLICIT_H) if atom.HasProp(
            self.IMPLICIT_H) else 0
        return atom.GetDegree() + implicit_h_num

    def writeBonds(self):
        if not self.bonds:
            return

        self.data_fh.write(f"{self.BONDS.capitalize()}\n\n")
        for bond_id, (bond_type, id1, id2) in self.bonds.items():
            bond_type = self.used_bond_types.index(
                bond_type) if self.concise else bond_type
            self.data_fh.write(f"{bond_id} {bond_type} {id1} {id2}\n")
        self.data_fh.write(f"\n")

    def getMatchesFromEnds(self, atoms, partial_matches, rough=False):
        o_symbols = set((
            x.GetSymbol(),
            self.getAtomConnt(x),
        ) for x in [atoms[0], atoms[-1]])
        ff_atom_ids = [
            [x, x.id1, x.id4] if hasattr(x, 'id4') else [x, x.id1, x.id3]
            for x in partial_matches
        ]
        ff_symbols = {
            x[0]: set([(
                self.ff.atoms[y].symbol,
                self.ff.atoms[y].connectivity,
            ) for y in x[1:]])
            for x in ff_atom_ids
        }
        matches = [x for x, y in ff_symbols.items() if y == o_symbols]
        if not matches:
            o_symbols_partial = set(x[0] for x in o_symbols)
            matches = [
                x for x, y in ff_symbols.items()
                if set(z[0] for z in y) == o_symbols_partial
            ]
        return matches

    def writeAngles(self):
        if not self.angles:
            return
        self.data_fh.write(f"{self.ANGLES.capitalize()}\n\n")
        angle_id = 0
        for _, (type_id, id1, id2, id3) in self.angles.items():
            angle_id += 1
            angle_type = self.used_angle_types.index(
                type_id) if self.concise else type_id
            self.data_fh.write(f"{angle_id} {angle_type} {id1} {id2} {id3}\n")
        self.data_fh.write(f"\n")

    def getDihedralAtoms(self, atom):
        dihe_atoms = []
        atomss = self.ff.getAngleAtoms(atom)
        atomss += [x[::-1] for x in atomss]
        for satom, matom, eatom in atomss:
            eatomss = self.ff.getAngleAtoms(eatom)
            matom_id = matom.GetIdx()
            eatom_id = eatom.GetIdx()
            for eatoms in eatomss:
                eatom_ids = [x.GetIdx() for x in eatoms]
                eatom_ids.remove(eatom_id)
                try:
                    eatom_ids.remove(matom_id)
                except ValueError:
                    continue
                dihe_4th = [x for x in eatoms if x.GetIdx() == eatom_ids[0]][0]
                dihe_atoms.append([satom, matom, eatom, dihe_4th])
        return dihe_atoms

    def getDihedralAtomsFromMol(self, mol):
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

    def setDihedrals(self):
        dihedral_id = 0
        for mol in self.mols.values():
            atomss_no_flip = self.getDihedralAtomsFromMol(mol)
            for atoms in atomss_no_flip:
                dihedral_id += 1
                matches = self.getMatchedDihedrals(atoms)
                dihedral = matches[0]
                self.dihedrals[dihedral_id] = (dihedral.id, ) + tuple(
                    [x.GetIntProp(self.ATOM_ID) for x in atoms])

    def getMatchedDihedrals(self, atoms):
        type_ids = [x.GetIntProp(self.TYPE_ID) for x in atoms]
        type_ids = [OPLS_Parser.DIHE_ATOM[x] for x in type_ids]
        if type_ids[1] > type_ids[2]:
            type_ids = type_ids[::-1]
        matches = [
            x for x in self.ff.dihedrals.values()
            if type_ids == [x.id1, x.id2, x.id3, x.id4]
        ]
        if matches:
            return matches
        partial_matches = [
            x for x in self.ff.dihedrals.values()
            if x.id2 == type_ids[1] and x.id3 == type_ids[2]
        ]
        if not partial_matches:
            rpm_ids = OPLS_Parser.DIHE_ATOMS[tuple(type_ids[1:3])]
            partial_matches = [
                x for x in self.ff.dihedrals.values()
                if set([x.id2, x.id3]) == set(rpm_ids)
            ]
        if not partial_matches:
            raise ValueError(
                f"No params for dihedral (middle bonded atom types {type_ids[1]}~{type_ids[2]})."
            )
        matches = self.getMatchesFromEnds(atoms, partial_matches)
        if not matches:
            raise ValueError(
                f"Cannot find params for dihedral between atom {'~'.join(map(str, type_ids))}."
            )
        return matches

    def writeDihedrals(self):
        if not self.dihedrals:
            return

        self.data_fh.write(f"{self.DIHEDRALS.capitalize()}\n\n")
        for dihedral_id, (type_id, id1, id2, id3,
                          id4) in self.dihedrals.items():
            type_id = self.used_dihedral_types.index(
                type_id) if self.concise else type_id
            self.data_fh.write(
                f"{dihedral_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_fh.write(f"\n")

    def setImpropers(self, symbols='CN', print_impropers=False):
        improper_id = 0
        for mol in self.mols.values():
            for atom in mol.GetAtoms():
                atom_symbol = atom.GetSymbol()
                if atom_symbol not in symbols:
                    continue
                neighbors = atom.GetNeighbors()
                # FIXME: H-N should be counted as one neighbor
                if atom.GetSymbol() not in 'CN' or len(neighbors) != 3:
                    continue
                if atom.GetSymbol() == 'N' and atom.GetHybridization(
                ) == Chem.rdchem.HybridizationType.SP3:
                    continue
                # Sp2 carbon for planar, Sp3 with one H (CHR1R2R3) for chirality,
                # Sp2 N in Amino Acid
                improper_id += 1
                neighbor_symbols = [x.GetSymbol() for x in neighbors]
                counted = self.countSymbols([atom_symbol] + neighbor_symbols)
                if print_impropers:
                    for symb, improper_ids in self.symbol_impropers.items():
                        print(f"{symb} {self.ff.impropers[improper_ids[0]]}")
                        impropers = [
                            self.ff.impropers[x] for x in improper_ids
                        ]
                        for improper in impropers:
                            print(
                                f"{[self.ff.atoms[x].description for x in [improper.id1, improper.id2, improper.id3, improper.id4]]}"
                            )
                improper_type_id = self.symbol_impropers[counted][0]
                neighbors = sorted(neighbors,
                                   key=lambda x: len(x.GetNeighbors()))
                for neighbor in neighbors:
                    if neighbor.GetSymbol(
                    ) == 'O' and neighbor.GetHybridization(
                    ) == Chem.rdchem.HybridizationType.SP2:
                        neighbors.remove(neighbor)
                        neighbors = [neighbor] + neighbors
                atoms = neighbors[:2] + [atom] + neighbors[2:]
                self.impropers[improper_id] = (improper_type_id, ) + tuple(
                    x.GetIntProp(self.ATOM_ID) for x in atoms)

    def AnglesByImpropers(self):

        for idx, (itype, id1, id2, id3, id4) in self.impropers.items():
            id124 = set([id1, id2, id4])
            aidxs = [
                aidx
                for aidx, (atype, aid1, aid2, aid3) in self.angles.items()
                if len(id124.intersection([aid1, aid3])) and id3 == aid2
            ]
            if len(aidxs) != 3:
                continue
            self.angles.pop(aidxs[2])

    def writeImpropers(self):

        if not self.impropers:
            return

        self.data_fh.write(f"{self.IMPROPERS.capitalize()}\n\n")
        for improper_id, (type_id, id1, id2, id3,
                          id4) in self.impropers.items():
            type_id = self.used_improper_types.index(
                type_id) if self.concise else type_id
            self.data_fh.write(
                f"{improper_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_fh.write(f"\n")

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

        :param symbols:
        :param csmbls:
        :return:
        """
        return ''.join((symbols[0], ) + tuple(y + str(symbols[1:].count(y))
                                              for y in csmbls))


class DataFileReader(LammpsWriter):

    SCALE = 0.6

    def __init__(self, data_file, min_dist=1.09 * 2):
        self.data_file = data_file
        self.min_dist = min_dist
        self.lines = None
        self.vdws = {}
        self.radii = {}
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.mols = {}
        self.excluded = collections.defaultdict(set)

    def run(self):
        self.read()
        self.setDescription()
        self.setAtoms()
        self.setBonds()
        self.setAngles()
        self.setDihedrals()
        self.setImpropers()
        self.setMols()

    def read(self):
        with open(self.data_file, 'r') as df_fh:
            self.lines = df_fh.readlines()
            self.mk_idxes = {
                x: i
                for i, l in enumerate(self.lines) for x in self.MARKERS
                if l.startswith(x)
            }

    def setDescription(self):

        dsp_eidx = min(self.mk_idxes.values())
        self.struct_dsp = {
            y: int(self.lines[x].split(y)[0])
            for x in range(dsp_eidx) for y in self.STRUCT_DSP
            if y in self.lines[x]
        }
        self.dype_dsp = {
            y: int(self.lines[x].split(y)[0])
            for x in range(dsp_eidx) for y in self.TYPE_DSP
            if y in self.lines[x]
        }
        self.box_dsp = {
            y: [float(z) for z in self.lines[x].split(y)[0].split()]
            for x in range(dsp_eidx) for y in self.BOX_DSP
            if y in self.lines[x]
        }

    def setMols(self):
        mols = collections.defaultdict(list)
        for atom in self.atoms.values():
            mols[atom.mol_id].append(atom.id)
        self.mols = dict(mols)

    def setAngles(self):
        sidx = self.mk_idxes[self.ANGLES_CAP] + 2
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.ANGLES]), 1):

            id, type_id, id1, id2, id3 = self.lines[lid].split()[:5]
            self.angles[int(id)] = types.SimpleNamespace(id=int(id),
                                                         type_id=int(type_id),
                                                         id1=int(id1),
                                                         id2=int(id2),
                                                         id3=int(id3))

    def setDihedrals(self):
        sidx = self.mk_idxes[self.DIHEDRALS_CAP] + 2
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

    def setClashParams(self, include14=True, scale=SCALE):
        self.setClashExclusion(include14=not include14)
        self.setPairCoeffs()
        self.setVdwRadius(scale=scale)

    def setClashExclusion(self, include14=True):
        pairs = [[x.id1, x.id2] for x in self.bonds.values()]
        pairs += [[x.id1, x.id3] for x in self.angles.values()]
        pairs += [
            y for x in self.impropers.values()
            for y in itertools.combinations([x.id1, x.id2, x.id3, x.id4], 2)
        ]
        if include14:
            pairs += [[x.id1, x.id4] for x in self.dihedrals.values()]
        for id1, id2 in pairs:
            self.excluded[id1].add(id2)
            self.excluded[id2].add(id1)

    def setPairCoeffs(self):
        sidx = self.mk_idxes[self.PAIR_COEFFS] + 2
        for lid in range(sidx, sidx + self.dype_dsp[self.ATOM_TYPES]):
            id, ene, dist = self.lines[lid].split()
            self.vdws[int(id)] = types.SimpleNamespace(id=int(id),
                                                       dist=float(dist),
                                                       ene=float(ene))

    def setVdwRadius(self, mix=LammpsWriter.GEOMETRIC, scale=1.):
        radii = collections.defaultdict(dict)
        for id1, vdw1 in self.vdws.items():
            for id2, vdw2 in self.vdws.items():
                if mix == LammpsWriter.GEOMETRIC:
                    dist = pow(vdw1.dist * vdw2.dist, 0.5)
                elif mix == LammpsWriter.ARITHMETIC:
                    dist = (vdw1.dist + vdw2.dist) / 2
                elif mix == LammpsWriter.SIXTHPOWER:
                    dist = pow((pow(vdw1.dist, 6) + pow(vdw2.dist, 6)) / 2,
                               1 / 6)
                dist *= pow(2, 1 / 6) * scale
                if dist < self.min_dist:
                    dist = self.min_dist
                radii[id1][id2] = dist

        self.radii = collections.defaultdict(dict)
        for atom1 in self.atoms.values():
            for atom2 in self.atoms.values():
                self.radii[atom1.id][atom2.id] = radii[atom1.type_id][
                    atom2.type_id]
        self.radii = dict(self.radii)


def main(argv):
    opls_parser = OPLS_Parser()
    opls_parser.read()


if __name__ == "__main__":
    main(sys.argv[1:])
