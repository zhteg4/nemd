import math
import copy
import sys
import argparse
import logutils
import os
import sys
import units
import parserutils
import fileutils
import nemd
import plotutils
import environutils
import jobutils
import symbols
import numpy as np
import opls
from rdkit import Chem

FlAG_CRU = 'cru'
FlAG_CRU_NUM = '-cru_num'
FlAG_BOND_LEN = '-bond_len'
FLAG_BOND_ANG = '-bond_ang'

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
                        type=parserutils.type_monomer_smiles,
                        help='')
    parser.add_argument(FlAG_CRU_NUM,
                        metavar=FlAG_CRU_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        help='')

    parser.add_argument(
        FlAG_BOND_LEN,
        metavar=FlAG_BOND_LEN[1:].upper(),
        type=parserutils.type_positive_float,
        default=1.5350,  # length of the C-H bond
        help='')
    parser.add_argument(
        FLAG_BOND_ANG,
        metavar=FLAG_BOND_ANG[1:].upper(),
        type=parserutils.type_positive_float,
        default=109.5,  # Tetrahedronal angle (C-C-C angle)
        help='')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


class Polymer(object):

    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.outfile = self.jobname + MOLT_OUT_EXT
        self.polym = None

    def run(self):
        self.polymerize()
        self.assignAtomType()
        log('Finished', timestamp=True)

    def setBondProj(self):
        bond_ang = self.options.bond_ang / 2. / 180. * math.pi
        bond_proj = self.options.bond_len * math.sin(bond_ang)
        pass

    def polymerize(self):
        mols = [
            copy.copy(self.options.cru) for x in range(self.options.cru_num)
        ]
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
        self.polym = Chem.DeleteSubstructs(polym, Chem.MolFromSmiles('*'))
        log(f"{Chem.MolToSmiles(self.polym)}")

    def assignAtomType(self):

        polym_smile = Chem.CanonSmiles(Chem.MolToSmiles(self.polym))
        for opls_mol in opls.OPLS_Parser.OPLSUA_MOLS:
            if polym_smile == opls_mol.smiles:
                for atom, atom_type_id in zip(self.polym.GetAtoms(),
                                              opls_mol.map):
                    atom.SetIntProp(opls.TYPE_ID, atom_type_id)
                return

        for opls_frag in opls.OPLS_Parser.OPLSUA_FRAGS:
            frag = Chem.MolFromSmiles(opls_frag.smiles)
            matches = self.polym.GetSubstructMatches(frag)
            for match in matches:
                for atom_id, type_id in zip(match, opls_frag.map):
                    if not type_id:
                        continue
                    atom = self.polym.GetAtoms()[atom_id]
                    try:
                        atom.GetIntProp(opls.TYPE_ID)
                    except KeyError:
                        atom.SetIntProp(opls.TYPE_ID, type_id)
                    else:
                        continue
                    log_debug(f"{atom_id} {type_id}")


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
