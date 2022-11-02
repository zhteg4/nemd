import os
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

# OPLSUA_MOLS = [
#     OPLSUA(smiles='C', map=(81, ), comment='CH4 Methane'),
#     # OPLSUA(smiles='C', map=(1,), comment='CH4 Methane'),
# ]

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
              UA(sml='CCCC', mp=(83, 86, 86, 83,), hs=None, dsc='n-Butane'),
              UA(sml='CC(C)C', mp=(84, 88, 84, 84, ), hs=None, dsc='Isobutane'),
              UA(sml='CC=CC', mp=(84, 89, 89, 84, ), hs=None, dsc='2-Butene'),
              # "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
              UA(sml='O=CO', mp=(134, 133, 135), hs={135: 136}, dsc='Carboxylic Acid'),
              # "Methyl", "=O Carboxylic Acid", "C Carboxylic Acid" , "-O- Carboxylic Acid"
              UA(sml='CC(=O)O', mp=(137, 133, 134, 135), hs={135: 136}, dsc='Ethanoic acid')]
    # yapf: enable
    SMILES = reversed(SMILES)
    ATOM_TOTAL = {i: i for i in range(1, 214)}
    BOND_ATOM = ATOM_TOTAL.copy()
    # "O Peptide Amide" "COH (zeta) Tyr" "OH Tyr"  "H(O) Ser/Thr/Tyr"
    BOND_ATOM.update({134: 2, 133: 26, 135: 23, 136: 24})
    ANGLE_ATOM = ATOM_TOTAL.copy()
    ANGLE_ATOM.update({134: 2, 133: 17, 135: 76, 136: 24})
    DIHE_ATOM = ATOM_TOTAL.copy()
    DIHE_ATOM.update({134: 11, 133: 26, 135: 76, 136: 24})

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
            ene_ang_ns = [
                ENE_ANG_N(ene=float(x), angle=float(y), n_parm=int(z))
                for x, y, z in zip(enes[::3], enes[1::3], enes[2::3])
            ]
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
    BOND_ATM_ID = 'bond_atm_id'
    IMPLICIT_H = 'implicit_h'

    ATOMS = 'atoms'
    BONDS = 'bonds'
    ANGLES = 'angles'
    DIHEDRALS = 'dihedrals'
    IMPROPERS = 'impropers'

    ATOM_TYPES = 'atom types'
    BOND_TYPES = 'bond types'
    ANGLE_TYPES = 'angle types'
    DIHEDRAL_TYPES = 'dihedral types'
    IMPROPER_TYPES = 'improper types'

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]

    MASSES = 'Masses'
    PAIR_COEFFS = 'Pair Coeffs'
    BOND_COEFFS = 'Bond Coeffs'
    ANGLE_COEFFS = 'Angle Coeffs'
    DIHEDRAL_COEFFS = 'Dihedral Coeffs'
    IMPROPER_COEFFS = 'Improper Coeffs'

    LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
    LJ_CUT = 'lj/cut'

    def __init__(self, ff, jobname, mols=None, lj_cut=11., coul_cut=11.):
        self.ff = ff
        self.jobname = jobname
        self.mols = mols
        self.lj_cut = lj_cut
        self.coul_cut = coul_cut
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
        self.pair_modify = {'mix': 'geometric'}
        self.special_bonds = {
            'lj/coul': (
                0.0,
                0.0,
                0.5,
            )
        }
        self.kspace_style = {'pppm': 0.0001}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.symbol_impropers = {}
        self.is_debug = environutils.is_debug()

    def writeLammpsIn(self):
        with open(self.lammps_in, 'w') as fh:
            fh.write(f"{self.UNITS} {self.units}\n")
            fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
            fh.write(f"{self.BOND_STYLE} {self.bond_style}\n")
            fh.write(f"{self.ANGLE_STYLE} {self.angle_style}\n")
            fh.write(f"{self.DIHEDRAL_STYLE} {self.dihedral_style}\n")
            fh.write(f"{self.IMPROPER_STYLE} {self.improper_style}\n")
            pair_style = self.LJ_CUT_COUL_LONG if self.hasCharge(
            ) else self.LJ_CUT
            fh.write(f"{self.PAIR_STYLE} {self.pair_style[pair_style]}\n")
            fh.write(
                f"{self.PAIR_MODIFY} {' '.join([(x,y) for x, y in self.pair_modify.items()][0])}\n"
            )
            special_bond = [
                f"{x} {' '.join(map(str, y))}"
                for x, y in self.special_bonds.items()
            ][0]
            fh.write(f"{self.SPECIAL_BONDS} {special_bond}\n")
            if self.hasCharge():
                kspace_style = [
                    f"{x} {y}" for x, y in self.kspace_style.items()
                ][0]
                fh.write(f"{self.KSPACE_STYLE} {kspace_style}\n")
            fh.write(f"{self.READ_DATA} {self.lammps_data}\n")

            fh.write("minimize 1.0e-4 1.0e-6 100 1000")

    def hasCharge(self, default=True):
        if self.mols is None:
            return default
        charges = [
            self.ff.charges[y.GetIntProp(self.TYPE_ID)]
            for x in self.mols.values() for y in x.GetAtoms()
        ]
        return any(charges)

    def writeLammpsData(self):

        with open(self.lammps_data, 'w') as self.data_fh:
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writePairCoeffs()
            self.writeBondCoeffs()
            self.writeAngleCoeffs()
            self.writeDihedralCoeffs()
            self.setImproperSymbols()
            self.writeImproperCoeffs()
            self.writeAtoms()
            self.setBonds()
            self.setAngles()
            self.setDihedrals()
            self.setImpropers()
            self.writeBonds()
            self.writeAngles()
            self.writeDihedrals()
            self.writeImpropers()

    def writeDescription(self):
        if self.mols is None:
            raise ValueError(f"Mols are not set.")

        self.data_fh.write(f"{self.LAMMPS_DESCRIPTION}\n\n")
        atoms = [len(x.GetAtoms()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(atoms)} {self.ATOMS}\n")
        bonds = [len(x.GetBonds()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(bonds)} {self.BONDS}\n")
        neighbors = [
            len(y.GetNeighbors()) for x in self.mols.values()
            for y in x.GetAtoms()
        ]
        # FIXME: I guess improper angles may reduce this num
        angles = [max(0, x - 1) for x in neighbors]
        self.data_fh.write(f"{sum(angles)} {self.ANGLES}\n")
        # FIXME: dihedral and improper are set to be zeros at this point
        self.data_fh.write(f"0 {self.DIHEDRALS}\n")
        self.data_fh.write(f"0 {self.IMPROPERS}\n\n")

    def writeTopoType(self):
        self.data_fh.write(f"{len(self.ff.atoms)} {self.ATOM_TYPES}\n")
        self.data_fh.write(f"{len(self.ff.bonds)} {self.BOND_TYPES}\n")
        self.data_fh.write(f"{len(self.ff.angles)} {self.ANGLE_TYPES}\n")
        self.data_fh.write(f"{len(self.ff.dihedrals)} {self.DIHEDRAL_TYPES}\n")
        self.data_fh.write(
            f"{len(self.ff.impropers)} {self.IMPROPER_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        if min_box is None:
            min_box = (20., 20., 20.,) # yapf: disable
        if buffer is None:
            buffer = (2., 2., 2.,) # yapf: disable
        xyzs = np.concatenate(
            [x.GetConformer(0).GetPositions() for x in self.mols.values()])
        box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
        box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]
        centroid = xyzs.mean(axis=0)
        for dim in range(3):
            self.data_fh.write(
                f"{centroid[dim]-box_hf[dim]:.2f} {centroid[dim]+box_hf[dim]:.2f} {self.LO_HI[dim]}\n"
            )
        self.data_fh.write("\n")

    def writeMasses(self):
        self.data_fh.write(f"{self.MASSES}\n\n")
        for atom_id, atom in self.ff.atoms.items():
            self.data_fh.write(f"{atom_id} {atom.mass} # {atom.description}\n")
        self.data_fh.write(f"\n")

    def writePairCoeffs(self):
        self.data_fh.write(f"{self.PAIR_COEFFS}\n\n")
        for atom in self.ff.atoms.values():
            vdw = self.ff.vdws[atom.id]
            self.data_fh.write(f"{atom.id} {atom.id} {vdw.ene} {vdw.dist}\n")
        self.data_fh.write("\n")

    def writeBondCoeffs(self):
        self.data_fh.write(f"{self.BOND_COEFFS}\n\n")
        for bond in self.ff.bonds.values():
            self.data_fh.write(f"{bond.id}  {bond.ene} {bond.dist}\n")
        self.data_fh.write("\n")

    def writeAngleCoeffs(self):
        self.data_fh.write(f"{self.ANGLE_COEFFS}\n\n")
        for angle in self.ff.angles.values():
            self.data_fh.write(f"{angle.id} {angle.ene} {angle.angle}\n")
        self.data_fh.write("\n")

    def writeDihedralCoeffs(self):
        self.data_fh.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        for dihedral in self.ff.dihedrals.values():
            params = [0., 0., 0., 0.]
            for ene_ang_n in dihedral.constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if (ene_ang_n.angle == 180.) ^ (ene_ang_n.n_parm in (
                        2,
                        4,
                )):
                    params[ene_ang_n.n_parm] *= -1
            self.data_fh.write(
                f"{dihedral.id}  {' '.join(map(str, params))}\n")
        self.data_fh.write("\n")

    def writeImproperCoeffs(self):
        self.data_fh.write(f"{self.IMPROPER_COEFFS}\n\n")
        for improper in self.ff.impropers.values():
            sign = 1 if improper.angle == 0. else -1
            self.data_fh.write(
                f"{improper.id} {improper.ene} {sign} {improper.n_parm}\n")
        self.data_fh.write("\n")

    def writeAtoms(self):
        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        atom_id = 0
        for mol_id, mol in self.mols.items():
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                atom_id += 1
                atom.SetIntProp(self.ATOM_ID, atom_id)
                type_id = atom.GetIntProp(self.TYPE_ID)
                xyz = conformer.GetAtomPosition(atom.GetIdx())
                xyz = ' '.join(map(lambda x: f'{x:.3f}', xyz))
                charge = self.ff.charges[type_id]
                dsrptn = self.ff.atoms[type_id].description
                self.data_fh.write(
                    f"{atom_id} {mol_id} {type_id} {charge} {xyz} # {dsrptn}\n"
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
        bond_utype = {
            bond: [
                set([bond.id1, bond.id2]).difference([mtype]).pop(),
                type_set.difference([mtype]).pop()
            ]
            for bond, mtype in partial_matches.items()
        }
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
            import pdb
            pdb.set_trace()
            raise ValueError(
                f"No params for angle (middle atom type {type_ids[1]}).")
        o_symbols = set([(
            x.GetSymbol(),
            self.getAtomConnt(x),
        ) for x in atoms[::2]])
        ff_symbols = {
            x: set([(
                self.ff.atoms[y].symbol,
                self.ff.atoms[y].connectivity,
            ) for y in [x.id1, x.id3]])
            for x in partial_matches
        }
        matches = [x for x, y in ff_symbols.items() if y == o_symbols]
        if not matches:
            o_symbols = set(x[0] for x in o_symbols)
            matches = [
                x for x, y in ff_symbols.items()
                if set(z[0] for z in y) == o_symbols
            ]
        if not matches:
            raise ValueError(
                f"No params for angle between atom {', '.join(map(str, type_ids))}."
            )
        self.debugPrintReplacement(atoms, matches)
        return matches

    def writeAngles(self):
        if not self.angles:
            return
        self.data_fh.write(f"{self.ANGLES.capitalize()}\n\n")
        for angle_id, (type_id, id1, id2, id3) in self.angles.items():
            self.data_fh.write(f"{angle_id} {type_id} {id1} {id2} {id3}\n")
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
                type_ids = [x.GetIntProp(self.TYPE_ID) for x in atoms]
                type_ids = [OPLS_Parser.DIHE_ATOM[x] for x in type_ids]
                if type_ids[1] > type_ids[2]:
                    type_ids = type_ids[::-1]
                matches = [
                    x for x in self.ff.dihedrals.values()
                    if type_ids == [x.id1, x.id2, x.id3, x.id4]
                ]
                if not matches:
                    raise ValueError(
                        f"Cannot find params for dihedral between atom {'~'.join(map(str, type_ids))}."
                    )
                dihedral = matches[0]
                self.dihedrals[dihedral_id] = (dihedral.id, ) + tuple(
                    [x.GetIntProp(self.ATOM_ID) for x in atoms])

    def writeDihedrals(self):
        if not self.dihedrals:
            return

        self.data_fh.write(f"{self.DIHEDRALS.capitalize()}\n\n")
        for dihedral_id, (type_id, id1, id2, id3,
                          id4) in self.dihedrals.items():
            self.data_fh.write(
                f"{dihedral_id} {type_id} {id1} {id2} {id3} {id4}\n")
        self.data_fh.write(f"\n")

    def setImpropers(self, symbols='CN'):
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
                for symb, improper_ids in self.symbol_impropers.items():
                    print(f"{symb} {self.ff.impropers[improper_ids[0]]}")
                    impropers = [self.ff.impropers[x] for x in improper_ids]
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
                len(neighbors[0].GetNeighbors())
                self.impropers[improper_id] = (
                    improper_type_id,
                    atom.GetIntProp(self.ATOM_ID),
                ) + tuple([x.GetIntProp(self.ATOM_ID) for x in neighbors])

    def writeImpropers(self):

        if not self.impropers:
            return

        self.data_fh.write(f"{self.IMPROPERS.capitalize()}\n\n")
        for improper_id, (type_id, id1, id2, id3,
                          id4) in self.impropers.items():
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


def main(argv):
    opls_parser = OPLS_Parser()
    opls_parser.read()


if __name__ == "__main__":
    main(sys.argv[1:])
