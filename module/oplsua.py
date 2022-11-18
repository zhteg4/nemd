import collections
import os
import re
import sys
import types
import symbols
import itertools
import logutils
import fileutils
import environutils
import numpy as np
from rdkit import Chem
from collections import namedtuple

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

    if logger is None:
        return
    logger.debug(msg)


def get_opls_parser():
    opls_parser = OPLS_Parser()
    opls_parser.read()
    return opls_parser


class OPLS_Parser:

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

    WRITE_ONCE = 'write_once'
    CHARGE = 'charge'
    IN_CHARGES = 'In Charges'
    DO_NOT_UA = 'DON\'T USE(OPLSUA)'

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
    # To get HO-C=O, COH~OH is used, which causes CH2-COOH bond issue
    BOND_ATOMS = {(26, 86): [16, 17], (26, 88): [16, 17]}
    # yapf: disable
    DIHE_ATOMS = {
        (26,86,): (1,6,), (26,88,): (1,6,)
    }
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

    def __init__(self, all_atom=False):
        self.all_atom = all_atom
        self.lines = None
        self.markers = None
        self.raw_content = {}
        self.content = {}
        self.atoms = {}
        self.vdws = {}
        self.bonds = {}
        self.angles = {}
        self.urey_bradleys = {}
        self.impropers = {}
        self.dihedrals = {}
        self.charges = {}
        self.mol_smiles = {}
        self.frag_smiles = {}

    def read(self):
        self.setRawContent()
        self.setAtomType()
        self.setVdW()
        self.setBond()
        self.setAngle()
        self.setUreyBradley()
        self.setImproper()
        self.setDihedral()
        self.setUACharge()

    def setRawContent(self):
        fp = open(self.FILE_PATH, 'r')
        lines = fp.readlines()
        lines = [x.strip(' \n') for x in lines]

        marker_lidx = {}
        for idx, line in enumerate(lines):
            for mark in self.MARKERS:
                if mark in line:
                    marker_lidx[mark] = idx

        for bmarker, emarker in zip(self.MARKERS[:-1], self.MARKERS[1:]):
            content_lines = lines[marker_lidx[bmarker]:marker_lidx[emarker]]
            self.raw_content[bmarker] = [
                x for x in content_lines
                if x and not x.startswith(symbols.POUND)
            ]

    def setAtomType(self):
        for line in self.raw_content[self.ATOM_MK]:
            bcomment, comment, acomment = line.split('"')
            _, id, formula = bcomment.split()
            atomic_number, mass, connectivity = acomment.split()
            h_count = 0
            symbol = formula.split('H')[0]
            if symbol:
                try:
                    h_count = int(formula.split('H')[-1])
                except ValueError:
                    # C as formula
                    h_count = int('H' in formula)
            else:
                # CH3 -> C; H -> H
                symbol = formula
            self.atoms[int(id)] = ATOM_TYPE(id=int(id),
                                            formula=formula,
                                            symbol=symbol,
                                            description=comment,
                                            atomic_number=int(atomic_number),
                                            mass=float(mass),
                                            connectivity=int(connectivity) +
                                            h_count)

    def setVdW(self):
        for line in self.raw_content[self.VAN_MK]:
            _, id, dist, ene = line.split()
            self.vdws[int(id)] = VDW(id=int(id),
                                     dist=float(dist),
                                     ene=float(ene))

    def setUACharge(self):
        for line in self.raw_content[self.ATOMIC_MK]:
            _, type_id, charge = line.split()
            self.charges[int(type_id)] = float(charge)

    def setBond(self):
        for id, line in enumerate(self.raw_content[self.BOND_MK], 1):
            _, id1, id2, ene, dist = line.split()
            self.bonds[id] = BOND(id=id,
                                  id1=int(id1),
                                  id2=int(id2),
                                  ene=float(ene),
                                  dist=float(dist))

    def setAngle(self):
        for id, line in enumerate(self.raw_content[self.ANGLE_MK], 1):
            _, id1, id2, id3, ene, angle = line.split()
            self.angles[id] = ANGLE(id=id,
                                    id1=int(id1),
                                    id2=int(id2),
                                    id3=int(id3),
                                    ene=float(ene),
                                    angle=float(angle))

    def setUreyBradley(self):
        for id, line in enumerate(self.raw_content[self.UREY_MK], 1):
            _, id1, id2, id3, ene, dist = line.split()
            self.urey_bradleys[id] = UREY_BRADLEY(id1=int(id1),
                                                  id2=int(id2),
                                                  id3=int(id3),
                                                  ene=float(ene),
                                                  dist=float(dist))

    def setImproper(self):
        for id, line in enumerate(self.raw_content[self.IMPROPER_MK], 1):
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
        for id, line in enumerate(self.raw_content[self.TORSIONAL_MK], 1):
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

    def setCharge(self):
        sidx = self.markers[self.IN_CHARGES]
        indexes = sorted(self.markers.values())
        eidx = indexes[indexes.index(sidx) + 1]
        lines = [
            x.strip().strip('set type') for x in self.lines[sidx + 1:eidx]
        ]
        lines = [
            x.strip(self.DO_NOT_UA).split(':')[1] for x in lines
            if self.all_atom ^ x.endswith(self.DO_NOT_UA)
        ]
        self.atoms = []
        for line in lines:
            idx_c, comment = line.split(symbols.POUND)
            index, charge = idx_c.split(self.CHARGE)
            atom = types.SimpleNamespace(index=int(index),
                                         charge=float(charge),
                                         comment=comment)
            self.atoms.append(atom)


class LammpsWriter(fileutils.LammpsInput):
    IN_EXT = '.in'
    DATA_EXT = '.data'
    LAMMPS_DESCRIPTION = 'LAMMPS Description'

    TYPE_ID = 'type_id'
    ATOM_ID = 'atom_id'
    RES_NUM = 'res_num'
    NEIGHBOR_CHARGE = 'neighbor_charge'
    BOND_ATM_ID = 'bond_atm_id'
    IMPLICIT_H = 'implicit_h'

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
    BUFFER = [
        4.,
        4.,
        4.,
    ]

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

    LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
    LJ_CUT = 'lj/cut'
    GEOMETRIC = 'geometric'
    ARITHMETIC = 'arithmetic'
    SIXTHPOWER = 'sixthpower'

    def __init__(self,
                 ff,
                 jobname,
                 mols=None,
                 lj_cut=11.,
                 coul_cut=11.,
                 timestep=1,
                 concise=True):
        self.ff = ff
        self.jobname = jobname
        self.mols = mols
        self.lj_cut = lj_cut
        self.coul_cut = coul_cut
        self.timestep = timestep
        self.concise = concise
        self.lammps_in = self.jobname + self.IN_EXT
        self.lammps_data = self.jobname + self.DATA_EXT
        self.units = 'real'
        self.atom_style = 'full'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'opls'
        self.improper_style = 'cvff'
        self.pair_style = {
            self.LJ_CUT_COUL_LONG:
            f"{self.LJ_CUT_COUL_LONG} {self.lj_cut} {self.coul_cut}",
            self.LJ_CUT: f"{self.LJ_CUT} {self.lj_cut}"
        }
        self.pair_modify = {'mix': self.GEOMETRIC}
        self.special_bonds = {
            'lj/coul': (
                0.0,
                0.0,
                0.5,
            )
        }
        self.kspace_style = {'pppm': 0.0001}
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
        self.in_fh = None
        self.data_fh = None
        self.is_debug = environutils.is_debug()

    def writeLammpsIn(self):
        with open(self.lammps_in, 'w') as self.in_fh:
            self.writeInDescriptions()
            self.readData()
            self.writeMinimize()
            self.writeTimestep()
            # self.writeRun()

    def writeInDescriptions(self):
        self.in_fh.write(f"{self.UNITS} {self.units}\n")
        self.in_fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
        self.in_fh.write(f"{self.BOND_STYLE} {self.bond_style}\n")
        self.in_fh.write(f"{self.ANGLE_STYLE} {self.angle_style}\n")
        self.in_fh.write(f"{self.DIHEDRAL_STYLE} {self.dihedral_style}\n")
        self.in_fh.write(f"{self.IMPROPER_STYLE} {self.improper_style}\n")
        pair_style = self.LJ_CUT_COUL_LONG if self.hasCharge() else self.LJ_CUT
        self.in_fh.write(f"{self.PAIR_STYLE} {self.pair_style[pair_style]}\n")
        self.in_fh.write(
            f"{self.PAIR_MODIFY} {' '.join([(x, y) for x, y in self.pair_modify.items()][0])}\n"
        )
        special_bond = [
            f"{x} {' '.join(map(str, y))}"
            for x, y in self.special_bonds.items()
        ][0]
        self.in_fh.write(f"{self.SPECIAL_BONDS} {special_bond}\n")
        if self.hasCharge():
            kspace_style = [f"{x} {y}"
                            for x, y in self.kspace_style.items()][0]
            self.in_fh.write(f"{self.KSPACE_STYLE} {kspace_style}\n")
        self.in_fh.write(f"log log.lammps\n")

    def hasCharge(self, default=True):
        if self.mols is None:
            return default
        charges = [
            self.ff.charges[y.GetIntProp(self.TYPE_ID)]
            for x in self.mols.values() for y in x.GetAtoms()
        ]
        return any(charges)

    def readData(self):
        self.in_fh.write(f"{self.READ_DATA} {self.lammps_data}\n")

    def writeMinimize(self, dump=True):
        if dump:
            self.in_fh.write(
                "dump 1 all custom 1000 dump.custom id xu yu zu\n")
            self.in_fh.write("dump_modify 1 sort id\n")
            atoms = self.ff.atoms.values()
            if self.concise:
                atoms = [x for x in atoms if x.id in self.used_atom_types]
            smbs = ' '.join(map(str, [x.symbol for x in atoms]))
            self.in_fh.write(f"dump_modify 1 element {smbs}\n")
        self.in_fh.write("minimize 1.0e-4 1.0e-6 100 1000\n")

    def writeTimestep(self):
        self.in_fh.write(f'timestep {self.timestep}\n')
        self.in_fh.write('thermo 1000\n')

    def writeRun(self):
        self.in_fh.write("velocity all create 10 482748\n")
        if len(self.mols) == 1 and self.mols[1].GetNumAtoms() < 10:
            # NVT on single molecule gives nan coords (guess due to translation)
            self.in_fh.write("fix             1 all nve\n")
            self.in_fh.write("run 10000\n")
            return

        self.in_fh.write(f"fix 1 all nvt temp 300 300 {self.timestep * 100}\n")
        self.in_fh.write("run 1000\n")

        if len(self.mols) == 1:
            return

        self.in_fh.write("unfix 1\n")
        self.in_fh.write(
            f"fix 1 all npt temp 10 10 {self.timestep * 100} iso 1 1 {self.timestep * 1000}\n"
        )
        self.in_fh.write("run 10000\n")

    def adjustConformer(self):
        self.setAtoms()
        self.setBonds()
        self.adjustBondLength()

    def writeLammpsData(self):

        with open(self.lammps_data, 'w') as self.data_fh:
            self.setImproperSymbols()
            self.setAtoms()
            self.setBonds()
            self.adjustBondLength()
            self.setAngles()
            self.setDihedrals()
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

    def removeUnused(self):
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
            box_hf = [max(box_hf) * 1.2 for x in box_hf]
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

    def setAtoms(self):
        atom_id = 0
        for mol_id, mol in self.mols.items():
            for atom in mol.GetAtoms():
                atom_id += 1
                atom.SetIntProp(self.ATOM_ID, atom_id)

    def adjustBondLength(self):
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

    def setBonds(self):
        bond_id = 0
        for mol in self.mols.values():
            for bond in mol.GetBonds():
                bond_id += 1
                bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
                bonded_atoms = sorted(
                    bonded_atoms, key=lambda x: x.GetIntProp(self.BOND_ATM_ID))
                matches = self.getMatchedBonds(bonded_atoms)
                bond = matches[0]
                self.bonds[bond_id] = (
                    bond.id,
                    bonded_atoms[0].GetIntProp(self.ATOM_ID),
                    bonded_atoms[1].GetIntProp(self.ATOM_ID),
                )

    def getMatchedBonds(self, bonded_atoms):
        """
        :param bonded_atoms: list of two bonded atoms sorted by BOND_ATM_ID
        :return:
        """
        atoms_types = [x.GetIntProp(self.BOND_ATM_ID) for x in bonded_atoms]
        try:
            atoms_types = OPLS_Parser.BOND_ATOMS[tuple(sorted(atoms_types))]
        except KeyError:
            # To get HO-C=O, COH~OH is used, which causes CH2-COOH bond issue
            pass
        # Exact match between two atom type ids
        matches = [
            x for x in self.ff.bonds.values() if [x.id1, x.id2] == atoms_types
        ]
        if matches:
            return matches

        log_debug(
            f"No exact params for bond between atom type {atoms_types[0]} and {atoms_types[1]}."
        )
        type_set = set(atoms_types)
        partial_matches = {
            x: type_set.intersection([x.id1, x.id2])
            for x in self.ff.bonds.values()
        }
        # {ff bond: one share atom type}
        partial_matches = {x: y.pop() for x, y in partial_matches.items() if y}
        bond_utype = {}
        for bond, mtype in partial_matches.items():
            bond_unmatched = set([bond.id1, bond.id2]).difference([mtype])
            bond_unmatched = bond_unmatched.pop() if bond_unmatched else mtype
            type_unmatched = type_set.difference([mtype])
            type_unmatched = type_unmatched.pop() if type_unmatched else mtype
            bond_utype[bond] = [bond_unmatched, type_unmatched]
        # ff bond: [unmatched atom type in ff bond, replaced unmatched atom type in ff, unmatched atom]
        bond_utype = {
            bond: [
                utype, rtype,
                [
                    x for x in bonded_atoms
                    if x.GetIntProp(self.BOND_ATM_ID) == rtype
                ][0]
            ]
            for bond, (utype, rtype) in bond_utype.items()
        }
        bond_score = {}
        for bond, (uatm, _, atm) in bond_utype.items():
            ssymbol = self.ff.atoms[uatm].symbol == atm.GetSymbol()
            scnnt = self.ff.atoms[uatm].connectivity == self.getAtomConnt(atm)
            bond_score[bond] = [ssymbol, scnnt]
        symbol_matched = [x for x, (y, z) in bond_score.items() if y]
        smbl_cnnt_matched = [x for x, y_z in bond_score.items() if all(y_z)]
        matches = smbl_cnnt_matched if smbl_cnnt_matched else symbol_matched
        if not matches:
            raise ValueError(
                f"No params for bond between atom type {atoms_types[0]} and {atoms_types[1]}."
            )
        self.debugPrintReplacement(bonded_atoms, matches)
        return matches

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

    def setAngles(self):
        angle_id = 0
        for mol in self.mols.values():
            for atom in mol.GetAtoms():
                for atoms in self.getAngleAtoms(atom):
                    angle_id += 1
                    matches = self.getMatchedAngles(atoms)
                    angle = matches[0]
                    self.angles[angle_id] = (angle.id, ) + tuple(
                        x.GetIntProp(self.ATOM_ID) for x in atoms)

    def getMatchedAngles(self, atoms):
        type_ids = [x.GetIntProp(self.TYPE_ID) for x in atoms]
        type_ids = [OPLS_Parser.ANGLE_ATOM[x] for x in type_ids]
        matches = [
            x for x in self.ff.angles.values()
            if type_ids == [x.id1, x.id2, x.id3]
        ]
        if matches:
            return matches
        log_debug(
            f"No exact params for angle between atom {', '.join(map(str, type_ids))}."
        )
        partial_matches = [
            x for x in self.ff.angles.values() if x.id2 == type_ids[1]
        ]
        if not partial_matches:
            raise ValueError(
                f"No params for angle (middle atom type {type_ids[1]}).")
        matches = self.getMatchesFromEnds(atoms, partial_matches)
        if not matches:
            raise ValueError(
                f"No params for angle between atom {', '.join(map(str, type_ids))}."
            )
        self.debugPrintReplacement(atoms, matches)
        return matches

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

    def getAngleAtoms(self, atom):
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            return []
        neighbors = sorted(neighbors, key=lambda x: x.GetIntProp(self.TYPE_ID))
        return [[x, atom, y] for x, y in itertools.combinations(neighbors, 2)]

    def getDihedralAtoms(self, atom):
        dihe_atoms = []
        atomss = self.getAngleAtoms(atom)
        atomss += [x[::-1] for x in atomss]
        for satom, matom, eatom in atomss:
            eatomss = self.getAngleAtoms(eatom)
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
                # Sp2 carbon for planar, Sp3 with one H (CHR1R2R3) for chirality, Sp2 N in Amino Acid
                improper_id += 1
                neighbor_symbols = [x.GetSymbol() for x in neighbors]
                counted = self.getCountedSymbols([atom_symbol] +
                                                 neighbor_symbols)
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

        symbolss = {
            z: ''.join([
                self.ff.atoms[y].symbol for y in [x.id1, x.id2, x.id3, x.id4]
            ])
            for z, x in self.ff.impropers.items()
        }
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
        log_debug(f"Impropers from the same symbols share the same constants.")
        orig_types = len(symbol_impropers)
        neighbors = [[x[2], x[0], x[1], x[3]] for x in symbol_impropers.keys()]
        # The csmbls in getCountedSymbols is from the following collections
        csmbls = sorted(set([y for x in neighbors[1:] for y in x]))
        counted = [(x[0], ) + tuple(y + str(x[1:].count(y)) for y in csmbls)
                   for x in neighbors]
        assert orig_types == len(counted)
        log_debug(f"Impropers neighbor counts based on symbols are unique.")
        for id, (symbols, constants) in enumerate(symbol_impropers.items()):
            counted_symbols = ''.join(counted[id])
            log_debug(f"{counted_symbols} ({symbols}) : {constants}")
            self.symbol_impropers[counted_symbols] = constants[3:]

    def getCountedSymbols(self, symbols, csmbls='CHNO'):
        return ''.join((symbols[0], ) + tuple(y + str(symbols[1:].count(y))
                                              for y in csmbls))


class DataFileReader(LammpsWriter):

    def __init__(self, data_file):
        self.data_file = data_file
        self.lines = None
        self.vdws = {}
        self.radii = {}
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.mols = {}
        self.excluded = collections.defaultdict(set)

    def run(self):
        self.read()
        self.setDescription()
        self.setAtoms()
        self.setBonds()
        self.setAngles()
        self.setDihedrals()
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

    def setAtoms(self):
        sidx = self.mk_idxes[self.ATOMS_CAP] + 2
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.ATOMS]), 1):
            id, mol_id, type_id, charge, x, y, z = self.lines[lid].split()[:7]
            ele = self.lines[lid].split('#')[-1].split()[-1]
            self.atoms[int(id)] = types.SimpleNamespace(
                id=int(id),
                mol_id=int(mol_id),
                type_id=int(type_id),
                charge=float(charge),
                xyz=[float(x), float(y), float(z)],
                ele=ele)

    def setMols(self):
        mols = collections.defaultdict(list)
        for atom in self.atoms.values():
            mols[atom.mol_id].append(atom.id)
        self.mols = dict(mols)

    def setBonds(self):
        sidx = self.mk_idxes[self.BONDS_CAP] + 2
        for id, lid in enumerate(
                range(sidx, sidx + self.struct_dsp[self.BONDS]), 1):
            id, type_id, id1, id2 = self.lines[lid].split()[:4]
            self.bonds[int(id)] = types.SimpleNamespace(id=int(id),
                                                        type_id=int(type_id),
                                                        id1=int(id1),
                                                        id2=int(id2))

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

    def setClashParams(self, include14=True, scale=1.):
        self.setClashExclusion(include14=include14)
        self.setPairCoeffs()
        self.setVdwRadius(scale=scale)

    def setClashExclusion(self, include14=True):
        pairs = [[x.id1, x.id2] for x in self.bonds.values()]
        pairs += [[x.id1, x.id3] for x in self.angles.values()]
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
                radii[id1][id2] = dist * pow(2, 1 / 6) * scale

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
