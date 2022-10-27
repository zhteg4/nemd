import os
import sys
import types
import symbols
import fileutils
from rdkit import Chem
from collections import namedtuple

OPLSUA = namedtuple('OPLSUA', ['smiles', 'map', 'comment'])

TYPE_ID = 'type_id'

OPLSUA_MOLS = [
    OPLSUA(smiles='C', map=(81, ), comment='CH4 Methane'),
    # OPLSUA(smiles='C', map=(1,), comment='CH4 Methane'),
]


class OPLS_Parser:

    FILE_PATH = fileutils.get_ff()
    WRITE_ONCE = 'write_once'
    CHARGE = 'charge'
    IN_CHARGES = 'In Charges'
    DO_NOT_UA = 'DON\'T USE(OPLSUA)'

    OPLSUA_MOLS = [
        OPLSUA(smiles='CC(=O)O', map=(
            6,
            3,
            4,
            5,
            7,
        ), comment='Acetic Acid'),
        OPLSUA(smiles='CC', map=(
            10,
            10,
        ), comment='Ethane'),
        OPLSUA(smiles='C', map=(10, ), comment='Methane')
    ]
    OPLSUA_FRAGS = [
        OPLSUA(smiles='CC(=O)O',
               map=(
                   None,
                   3,
                   4,
                   5,
                   7,
               ),
               comment='Acetic Acid'),
        OPLSUA(smiles='CCC', map=(
            None,
            13,
            None,
        ), comment='Alkanes'),
        OPLSUA(smiles='CCC', map=(10, None, 10), comment='Alkanes')
    ]

    def __init__(self, all_atom=False):
        self.all_atom = all_atom
        self.lines = None
        self.markers = None
        self.atoms = None

    def read(self):
        fp = open(self.FILE_PATH, 'r')
        lines = fp.readlines()
        lines = [x.strip() for x in lines]
        self.lines = [
            x for x in lines
            if x and not (x.startswith(symbols.POUND) or x == symbols.RETURN
                          or x.startswith(symbols.RC_BRACKET))
        ]
        self.markers = {
            x.strip(self.WRITE_ONCE).strip('(){} \"'): i
            for i, x in enumerate(self.lines) if x.startswith(self.WRITE_ONCE)
        }

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
        import pdb
        pdb.set_trace()
        pass


def main(argv):
    opls_parser = OPLS_Parser()
    opls_parser.read()
    opls_parser.setCharge()
    opls_parser.setSmiles()


if __name__ == "__main__":
    main(sys.argv[1:])