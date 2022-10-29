import os
import sys
import types
import symbols
import fileutils
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

UA = namedtuple('UA', ['sml', 'mp', 'dsc'])

TYPE_ID = fileutils.TYPE_ID

# OPLSUA_MOLS = [
#     OPLSUA(smiles='C', map=(81, ), comment='CH4 Methane'),
#     # OPLSUA(smiles='C', map=(1,), comment='CH4 Methane'),
# ]


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

    # OPLSUA_MOLS = [
    #     OPLSUA(smiles='CC(=O)O', map=(
    #         6,
    #         3,
    #         4,
    #         5,
    #         7,
    #     ), comment='Acetic Acid'),
    #     OPLSUA(smiles='CC', map=(
    #         10,
    #         10,
    #     ), comment='Ethane'),
    #     OPLSUA(smiles='C', map=(10, ), comment='Methane')
    # ]
    # OPLSUA_FRAGS = [
    #     OPLSUA(smiles='CC(=O)O',
    #            map=(
    #                None,
    #                3,
    #                4,
    #                5,
    #                7,
    #            ),
    #            comment='Acetic Acid'),
    #     OPLSUA(smiles='CCC', map=(
    #         None,
    #         13,
    #         None,
    #     ), comment='Alkanes'),
    #     OPLSUA(smiles='CCC', map=(10, None, 10), comment='Alkanes')
    # ]

    # yapf: disable
    SMILES = [UA(sml='C', mp=(81, ), dsc='CH4 Methane'),
              UA(sml='CC', mp=(82, 82,), dsc='Ethane'),
              UA(sml='CCCC', mp=(83, 86, 86, 83,), dsc='n-Butane'),
              UA(sml='CC(C)C', mp=(84, 88, 84, 84, ), dsc='Isobutane'),
              UA(sml='CC=CC', mp=(84, 89, 89, 84, ), dsc='2-Butene')]
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
            symbol = formula.split('H')[0]
            if not symbol:
                # CH3 -> C; H -> H
                symbol = formula
            self.atoms[int(id)] = ATOM_TYPE(id=int(id),
                                            formula=formula,
                                            symbol=symbol,
                                            description=comment,
                                            atomic_number=int(atomic_number),
                                            mass=float(mass),
                                            connectivity=int(connectivity))

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

    def setSmiles(self):
        # FIXME: manually match smiles with the comments and then
        # parse the *.lt to generate OPLSUA_MOLS
        # mass for element, neighbor atom and bond type to find matches
        # Chem.AddHs(mol)
        formulas = {
            x.description: x.formula
            for x in self.atoms.values() if x.connectivity == 1
        }
        import pdb
        pdb.set_trace()
        for formula in formulas:
            Chem.MolFromSmiles(self.FORMULA_SMILES[formula])
        import pdb
        pdb.set_trace()
        pass


def main(argv):
    opls_parser = OPLS_Parser()
    opls_parser.read()
    opls_parser.setSmiles()


if __name__ == "__main__":
    main(sys.argv[1:])
