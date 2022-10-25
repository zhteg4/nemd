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
import itertools
import plotutils
import environutils
import jobutils
import symbols
import numpy as np
import oplsua
from rdkit import Chem
from rdkit.Chem import AllChem

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
                        type=parserutils.type_monomer_smiles,
                        help='')
    parser.add_argument(FlAG_CRU_NUM,
                        metavar=FlAG_CRU_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        default=1,
                        help='')
    parser.add_argument(FlAG_MOL_NUM,
                        metavar=FlAG_MOL_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        default=1,
                        help='')

    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


class Polymer(object):
    ATOM_ID = 'atom_id'
    TYPE_ID = 'type_id'
    MOL_NUM = 'mol_num'

    def __init__(self, options, jobname, ff=None):
        self.options = options
        self.jobname = jobname
        self.ff = ff
        self.outfile = self.jobname + MOLT_OUT_EXT
        self.polym = None
        self.polym_Hs = None
        self.mols = {}
        self.buffer = [2., 2., 2.]
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()

    def run(self):
        self.polymerize()
        self.assignAtomType()
        self.embedMol()
        self.setMols()
        self.write()
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
        self.polym = Chem.DeleteSubstructs(
            polym, Chem.MolFromSmiles(symbols.WILD_CARD))
        log(f"{Chem.MolToSmiles(self.polym)}")

    def assignAtomType(self):
        polym_smile = Chem.CanonSmiles(Chem.MolToSmiles(self.polym))
        for sml in self.ff.SMILES:
            if polym_smile == sml.sml:
                for atom, atom_type_id in zip(self.polym.GetAtoms(), sml.mp):
                    atom.SetIntProp(self.TYPE_ID, atom_type_id)
                    log_debug(f"{atom.GetIdx()} {atom_type_id}")
                return

        for opls_frag in ff_frag:
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

    def embedMol(self):
        self.polym_Hs = Chem.AddHs(self.polym)
        AllChem.EmbedMolecule(self.polym_Hs)
        self.polym = Chem.RemoveHs(self.polym_Hs)

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
        lmw = fileutils.LammpsWriter(self.ff, self.jobname, mols=self.mols)
        lmw.writeLammpsIn()
        lmw.writeLammpsData()

        # with open(self.outfile, 'w') as fh:
        #     fh.write('import "oplsaa.lt"\n\n\n')
        #     fh.write("%s inherits OPLSAA {\n\n" % self.jobname.capitalize())
        #     fh.write(
        #         '# atomID   molID  atomTyle  charge     X        Y          Z\n'
        #     )
        #
        #     fh.write('write("Data Atoms") {\n')
        #     conformer = self.polym.GetConformer(0)
        #     atom_id = 0
        #     for atom in self.polym.GetAtoms():
        #         xyz = ' '.join(
        #             map(str, conformer.GetAtomPosition(atom.GetIdx())))
        #         fh.write(
        #             f"  $atom:{atom_id} $mol:. @atom:{atom.GetIntProp(self.TYPE_ID)} 0. {xyz}\n"
        #         )
        #         atom.SetIntProp(self.ATOM_ID, atom_id)
        #         atom_id += 1
        #     fh.write("}\n\n")
        #
        #     fh.write('write("Data Bond List") {\n')
        #     for id, bond in enumerate(self.polym.GetBonds()):
        #         batom_id = bond.GetBeginAtom().GetIntProp(self.ATOM_ID)
        #         eatom_id = bond.GetEndAtom().GetIntProp(self.ATOM_ID)
        #         fh.write(f"  $bond:{id} $atom:{batom_id} $atom:{eatom_id}\n")
        #     fh.write("}\n\n")
        #
        #     fh.write("}\n\n")


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
