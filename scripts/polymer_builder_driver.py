import os
import sys
import math
import copy
import oplsua
import lammps
import symbols
import jobutils
import logutils
import functools
import constants
import fragments
import structutils
import parserutils
import fileutils
import itertools
import prop_names
import collections
import environutils
import conformerutils
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

FlAG_CRU = 'cru'
FlAG_CRU_NUM = '-cru_num'
FlAG_MOL_NUM = '-mol_num'
FlAG_SEED = '-seed'

MOLT_OUT_EXT = fileutils.MOLT_FF_EXT

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: the msg to be printed
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(
        description='Build amorphous cell from molecules and monomers.')
    parser.add_argument(
        FlAG_CRU,
        metavar=FlAG_CRU.upper(),
        type=functools.partial(parserutils.type_monomer_smiles,
                               allow_mol=True),
        nargs='+',
        help='SMILES of the constitutional repeat unit (monomer)')
    parser.add_argument(
        FlAG_CRU_NUM,
        metavar=FlAG_CRU_NUM[1:].upper(),
        type=parserutils.type_positive_int,
        nargs='+',
        help='Number of constitutional repeat unit per polymer')
    parser.add_argument(FlAG_MOL_NUM,
                        metavar=FlAG_MOL_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        nargs='+',
                        help='Number of molecules in the amorphous cell')
    parser.add_argument(FlAG_SEED,
                        metavar=FlAG_SEED[1:].upper(),
                        type=parserutils.type_random_seed,
                        help='Set random state using this seed.')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


class AmorphousCell(object):
    """
    Build amorphous cell from molecules. (polymers may be built from monomers first)
    """

    def __init__(self, options, jobname, ff=None):
        """
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        :param jobname str: the jobname based on which out filenames are figured
            out
        :param ff str: the force field filepath
        """
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
        """
        Main method to build the cell.
        """
        self.setPolymers()
        self.setGriddedCell()
        self.write()

    def setPolymers(self):
        """
        Build polymer from monomers if provided.
        """
        for cru, cru_num, in zip(self.options.cru, self.options.cru_num):
            polym = Polymer(cru, cru_num)
            polym.run()
            self.polymers.append(polym)

    def setGriddedCell(self):
        """
        Set gridded amorphous cell.
        """
        self.setBoxes()
        self.setPolymVectors()
        self.placeMols()

    def setBoxes(self):
        """
        Set the minimum box for each molecule.
        """

        for polymer in self.polymers:
            xyzs = polymer.polym.GetConformer(0).GetPositions()
            polymer.box = xyzs.max(axis=0) - xyzs.min(axis=0) + polymer.buffer
        self.mbox = np.array([x.box for x in self.polymers]).max(axis=0)

    def setPolymVectors(self):
        """
        Set polymer translational vectors based on medium box size.
        """
        for polym, mol_num in zip(self.polymers, self.options.mol_num):
            polym.mol_num = mol_num
            mol_nums_per_mbox = np.floor(self.mbox / polym.box).astype(int)
            polym.mol_num_per_mbox = np.product(mol_nums_per_mbox)
            polym.num_mbox = math.ceil(polym.mol_num / polym.mol_num_per_mbox)
            percent = [
                np.linspace(-0.5, 0.5, x, endpoint=False)
                for x in mol_nums_per_mbox
            ]
            percent = [x - x.mean() for x in percent]
            polym.vecs = [
                x * self.mbox
                for x in itertools.product(*[[y for y in x] for x in percent])
            ]

    def placeMols(self):
        """
        Duplicate molecules and set coordinates.
        """
        idxs = range(
            math.ceil(math.pow(sum(x.num_mbox for x in self.polymers),
                               1. / 3)))
        vectors = [x * self.mbox for x in itertools.product(idxs, idxs, idxs)]
        mol_id, polymers = 0, self.polymers[:]
        while polymers:
            np.random.shuffle(vectors)
            vector = vectors.pop()
            polymer = np.random.choice(polymers)
            for idx in range(min([polymer.mol_num, polymer.mol_num_per_mbox])):
                polymer.mol_num -= 1
                if polymer.mol_num == 0:
                    polymers.remove(polymer)
                mol_id, mol = mol_id + 1, copy.copy(polymer.polym)
                self.mols[mol_id] = mol
                mol.SetIntProp(Polymer.MOL_NUM, mol_id)
                conformerutils.translation(mol.GetConformer(0),
                                           polymer.vecs[idx] + vector)

    def write(self):
        """
        Write amorphous cell into data file.
        :return:
        """
        lmw = oplsua.LammpsWriter(self.mols, self.ff, self.jobname)
        lmw.writeData(adjust_coords=False)
        lmw.writeLammpsIn()
        log(f'Data file written into {lmw.lammps_data}.')
        log(f'In script written into {lmw.lammps_in}.')


class Polymer(object):
    """
    Class to build a polymer from monomers.
    """

    ATOM_ID = oplsua.LammpsWriter.ATOM_ID
    TYPE_ID = oplsua.LammpsWriter.TYPE_ID
    BOND_ATM_ID = oplsua.LammpsWriter.BOND_ATM_ID
    RES_NUM = oplsua.RES_NUM
    NEIGHBOR_CHARGE = oplsua.LammpsWriter.NEIGHBOR_CHARGE
    MOL_NUM = 'mol_num'
    IMPLICIT_H = oplsua.LammpsWriter.IMPLICIT_H
    MONO_ATOM_IDX = 'mono_atom_idx'
    CAP = 'cap'
    HT = 'ht'
    POLYM_HT = prop_names.POLYM_HT
    IS_MONO = prop_names.IS_MONO
    MONO_ID = prop_names.MONO_ID

    def __init__(self, cru, cru_num, ff=None):
        """
        :param cru str: the smiles string for monomer
        :param cru_num int: the number of monomers per polymer
        :param ff str: the file path for the force field
        """
        self.cru = cru
        self.cru_num = cru_num
        self.ff = ff
        self.polym = None
        self.polym_Hs = None
        self.box = None
        self.cru_mol = None
        self.molecules = []
        self.buffer = oplsua.LammpsWriter.BUFFER
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()

    def run(self):
        """
        Main method to build one polymer.
        """
        self.setCruMol()
        self.markMonomer()
        self.polymerize()
        self.assignAtomType()
        self.embedMol()

    def setCruMol(self):
        """
        Set monomer mol based on the input smiles.
        """
        cru_mol = Chem.MolFromSmiles(self.cru)
        for atom in cru_mol.GetAtoms():
            if atom.GetSymbol() != symbols.CARBON or atom.GetIsAromatic():
                continue
            atom.SetIntProp(self.IMPLICIT_H, atom.GetNumImplicitHs())
            atom.SetNoImplicit(True)
        # FIXME: support monomers of different chiralties
        chiralty_info = Chem.FindMolChiralCenters(cru_mol,
                                                  includeUnassigned=True)
        for chiralty in chiralty_info:
            cru_mol.GetAtomWithIdx(chiralty[0]).SetProp('_CIPCode', 'R')

        self.cru_mol = Chem.AddHs(cru_mol)

    def markMonomer(self):
        """
        Marker monomer original atom indexes, head/tail and capping atoms.
        """

        capping = [
            x for x in self.cru_mol.GetAtoms()
            if x.GetSymbol() == symbols.WILD_CARD
        ]
        is_mono = True if capping else False
        self.cru_mol.SetBoolProp(self.IS_MONO, is_mono)
        [x.SetBoolProp(self.CAP, True) for x in capping]
        [
            y.SetBoolProp(self.HT, True) for x in capping
            for y in x.GetNeighbors()
        ]
        [
            x.SetIntProp(self.MONO_ATOM_IDX, x.GetIdx())
            for x in self.cru_mol.GetAtoms()
        ]

    def polymerize(self):
        """
        Polymerize from the monomer mol.
        """

        if not self.cru_mol.GetBoolProp(self.IS_MONO):
            self.polym = self.cru_mol
            return
        # Duplicate and index monomers
        mols = []
        for mono_id in range(1, self.cru_num + 1):
            mol = copy.copy(self.cru_mol)
            mols.append(mol)
            for atom in mol.GetAtoms():
                atom.SetIntProp(self.MONO_ID, mono_id)
        # Combine monomers into one molecule
        combo = mols[0]
        for mol in mols[1:]:
            combo = Chem.CombineMols(combo, mol)
        # Search head/tail atoms
        capping_atoms = [
            x for x in combo.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
        ]
        ht_atoms = [x.GetNeighbors()[0] for x in capping_atoms]
        ht_atom_idxs = [x.GetIdx() for x in ht_atoms]
        for polym_ht in [ht_atom_idxs[0], ht_atom_idxs[-1]]:
            atom = combo.GetAtomWithIdx(polym_ht)
            atom.SetBoolProp(self.POLYM_HT, True)
            atom.SetIntProp(self.IMPLICIT_H,
                            atom.GetIntProp(self.IMPLICIT_H) + 1)
        # Create bonds between monomers
        edcombo = Chem.EditableMol(combo)
        for t_atom_idx, h_atom_idx in zip(
                ht_atom_idxs[1:-1:2],
                ht_atom_idxs[2::2],
        ):
            edcombo.AddBond(t_atom_idx,
                            h_atom_idx,
                            order=Chem.rdchem.BondType.SINGLE)
        polym = edcombo.GetMol()
        # Delete capping atoms
        orgin_atom_num = None
        while (orgin_atom_num != polym.GetNumAtoms()):
            orgin_atom_num = polym.GetNumAtoms()
            polym = Chem.DeleteSubstructs(
                polym, Chem.MolFromSmiles(symbols.WILD_CARD))
        self.polym = polym
        log(f"Polymer SMILES: {Chem.MolToSmiles(self.polym)}")

    def assignAtomType(self):
        """
        Assign atom types to the structure.
        """
        ff_typer = oplsua.OPLS_Typer(self.polym)
        ff_typer.run()

    def embedMol(self, trans=False):
        """
        Embed the molecule with coordinates.

        :param trans bool: If True, all_trans conformer without entanglements is
            built.
        """
        if self.polym.GetNumAtoms() <= 200 and not trans:
            AllChem.EmbedMolecule(self.polym, useRandomCoords=True)
            return

        trans_conf = Conformer(self.polym, self.cru_mol, trans=trans)
        trans_conf.run()

    def write(self):
        """
        Write lammps data file.
        """
        lmw = oplsua.LammpsWriter(self.ff, self.jobname, mols=self.mols)
        lmw.writeData()
        lmw.writeLammpsIn()


class Conformer(object):
    """
    Conformer coordinate assignment.
    """

    CAP = Polymer.CAP
    MONO_ATOM_IDX = Polymer.MONO_ATOM_IDX
    MONO_ID = Polymer.MONO_ID
    OUT_EXTN = '.sdf'

    def __init__(self,
                 polym,
                 original_cru_mol,
                 ff=None,
                 trans=False,
                 jobname=None,
                 minimization=True):
        """
        :param polym 'rdkit.Chem.rdchem.Mol': the polymer to set conformer
        :param original_cru_mol 'rdkit.Chem.rdchem.Mol': the monomer mol
            constructing the polymer
        :param ff 'OPLS_Parser': force field information.
        :param trans bool: Whether all-tran conformation is requested.
        :param jobname str: The jobname
        :param minimization bool: Whether LAMMPS minimization is performed.
        """
        self.polym = polym
        self.original_cru_mol = original_cru_mol
        self.ff = ff
        self.trans = trans
        self.jobname = jobname
        self.minimization = minimization
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()
        if self.jobname is None:
            self.jobname = 'conf_search'
        self.relax_dir = '_relax'
        self.data_file = 'data.polym'
        self.conformer = None
        self.lmw = None

    def run(self):
        """
        Main method set the conformer.
        """

        self.setCruMol()
        self.setCruConformer()
        self.setCruBackbone()
        self.transAndRotate()
        self.rotateSideGroups()
        self.setXYZAndVect()
        self.setConformer()
        self.adjustConformer()
        self.minimize()
        self.foldPolym()

    def setCruMol(self):
        """
        Set monomer mol with elements tuned.
        """
        cru_mol = copy.copy(self.original_cru_mol)
        atoms = [
            x for x in cru_mol.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
        ]
        neighbors = [x.GetNeighbors()[0] for x in atoms[::-1]]
        for atom, catom in zip(atoms, neighbors):
            atom.SetAtomicNum(catom.GetAtomicNum())
            atom.SetBoolProp(self.CAP, True)
        self.cru_mol = cru_mol

    def setCruConformer(self):
        """
        Set the cru conformer.
        """
        AllChem.EmbedMolecule(self.cru_mol)
        self.cru_conformer = self.cru_mol.GetConformer(0)

    def setCruBackbone(self):
        """
        Set the cru backbone atom ids.
        """
        cap_idxs = [
            x.GetIdx() for x in self.cru_mol.GetAtoms() if x.HasProp(self.CAP)
        ]
        if len(cap_idxs) != 2:
            raise ValueError(f'{len(cap_idxs)} capping atoms are found.')
        graph = structutils.getGraph(self.cru_mol)
        self.cru_bk_atom_ids = nx.shortest_path(graph, *cap_idxs)

    def transAndRotate(self):
        """
        Set trans-conformer with translation and rotation.
        """

        for dihe in zip(self.cru_bk_atom_ids[:-3], self.cru_bk_atom_ids[1:-2],
                        self.cru_bk_atom_ids[2:-1], self.cru_bk_atom_ids[3:]):
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conformer, *dihe, 180)

        cntrd = conformerutils.centroid(self.cru_conformer,
                                        atom_ids=self.cru_bk_atom_ids)

        conformerutils.translation(self.cru_conformer, -np.array(cntrd))
        abc_norms = self.getABCVectors()
        abc_targeted = np.eye(3)
        conformerutils.rotate(self.cru_conformer, abc_norms, abc_targeted)

    def getABCVectors(self):
        """
        Get the a, b, c vectors of the molecule.

        :return 3x3 'numpy.ndarray': a vector is parallel with the backbone,
            b and c vectors are perpendicular with the backbone.
        """

        def get_norm(vect):
            """
            Get the normalized vector.

            :param vect 'numpy.ndarray': input vector
            :return 'numpy.ndarray': normalized vector
            """
            vect /= np.linalg.norm(vect)
            return vect

        bh_xyzs = np.array([
            self.cru_conformer.GetAtomPosition(x) for x in self.cru_bk_atom_ids
        ])
        bvectors = (bh_xyzs[1:, :] - bh_xyzs[:-1, :])
        nc_vector = get_norm(bvectors[::2].mean(axis=0))
        nm_mvector = get_norm(bvectors[1::2].mean(axis=0))
        avect = get_norm(nc_vector + nm_mvector).reshape(1, -1)
        bvect = get_norm(nc_vector - nm_mvector).reshape(1, -1)
        cvect = get_norm(np.cross(avect, bvect)).reshape(1, -1)
        abc_norms = np.concatenate([avect, bvect, cvect], axis=0)
        return abc_norms

    def rotateSideGroups(self):
        """
        Rotate the bond between the backbone and side group.
        """

        def get_other_atom(aid1, aid2):
            """
            Get one atom id that is bonded to aid1 beyond aid2.

            :param aid1 int: The neighbors of this atom is searched.
            :param aid2 int: This is excluded from the neighbor search
            :return one atom id that is bonded to aid1 beyond aid2:
            """
            aids = self.cru_mol.GetAtomWithIdx(aid1).GetNeighbors()
            aids = [x.GetIdx() for x in aids]
            aids.remove(aid2)
            return aids[0]

        # Search for atom pairs connecting the backbone and the side group
        bonds = self.cru_mol.GetBonds()
        bonded_aids = [(x.GetBeginAtomIdx(), x.GetEndAtomIdx()) for x in bonds]
        bk_ids = set(self.cru_bk_atom_ids)
        side_aids = [
            x for x in bonded_aids if len(bk_ids.intersection(x)) == 1
        ]
        side_aids = [x if x[0] in bk_ids else reversed(x) for x in side_aids]
        # Get the dihedral atom ids that move the side group
        side_dihes = []
        for id2, id3 in side_aids:
            id1 = get_other_atom(id2, id3)
            id4 = get_other_atom(id3, id2)
            side_dihes.append([id1, id2, id3, id4])
        for dihe in side_dihes:
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conformer, *dihe, 90)

    def setXYZAndVect(self):
        """
        Set the XYZ of the non-capping atoms and translational vector, preparing
        for the monomer coordinate shifting.
        """

        cap_ht = [x for x in self.cru_mol.GetAtoms() if x.HasProp(self.CAP)]
        cap_ht = [(x.GetIdx(), x.GetNeighbors()[0].GetIdx()) for x in cap_ht]
        middle_points = np.array([
            np.mean([self.cru_conformer.GetAtomPosition(y) for y in x], axis=0)
            for x in cap_ht
        ])

        self.vector = middle_points[1, :] - middle_points[0, :]
        self.xyzs = {
            x.GetIntProp(self.MONO_ATOM_IDX):
            np.array(self.cru_conformer.GetAtomPosition(x.GetIdx()))
            for x in self.cru_mol.GetAtoms() if x.HasProp(self.MONO_ATOM_IDX)
        }

    def setConformer(self):
        """
        Build and set the conformer.
        """
        mono_id_atoms = collections.defaultdict(list)
        for atom in self.polym.GetAtoms():
            mono_id_atoms[atom.GetIntProp(self.MONO_ID)].append(atom)

        conformer = Chem.rdchem.Conformer(self.polym.GetNumAtoms())
        for mono_id, atoms in mono_id_atoms.items():
            vect = mono_id * self.vector
            for atom in atoms:
                mono_atom_id = atom.GetIntProp(self.MONO_ATOM_IDX)
                xyz = self.xyzs[mono_atom_id] + vect
                conformer.SetAtomPosition(atom.GetIdx(), xyz)
        self.polym.AddConformer(conformer)
        Chem.GetSymmSSSR(self.polym)

    def adjustConformer(self):
        """
        Adjust the conformer coordinates based on the force field.
        """
        mols = {1: self.polym}
        self.lmw = oplsua.LammpsWriter(self.ff, self.jobname, mols=mols)
        if self.minimization:
            return
        self.lmw.adjustCoords()

    def minimize(self):
        """
        Run force field minimizer.
        """

        if not self.minimization:
            return

        with fileutils.chdir(self.relax_dir):
            self.lmw.writeData()
            self.lmw.writeLammpsIn()
            lmp = lammps.lammps(cmdargs=['-screen', 'none'])
            lmp.file(self.lmw.lammps_in)
            lmp.command(f'write_data {self.data_file}')
            lmp.close()
            # Don't delete: The following is one example for interactive lammps
            # min_cycle=10, max_cycle=100, threshold=.99
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
        """
        Fold polymer chain with clash check.
        """

        if self.trans or not self.minimization:
            return

        data_file = os.path.join(self.relax_dir, self.data_file)
        fmol = fragments.FragMol(self.polym, data_file=data_file)
        fmol.run()
        log('An entangled conformer set.')

    @classmethod
    def write(cls, mol, filename):
        """
        Write the polymer and monomer into sdf files.

        :param mol 'rdkit.Chem.rdchem.Mol': The molecule to write out
        :param filename str: The file path to write into
        """

        with Chem.SDWriter(filename) as fh:
            try:
                mono_atom_idxs = [
                    x.GetIntProp(cls.MONO_ATOM_IDX) for x in mol.GetAtoms()
                ]
            except KeyError:
                fh.write(mol)
                return
            mol.SetProps([cls.MONO_ATOM_IDX])
            mol.SetProp(cls.MONO_ATOM_IDX, ' '.join(map(str, mono_atom_idxs)))
            fh.write(mol)

    @classmethod
    def read(cls, filename):
        """
        Read molecule from file path.

        :param filename str: the file path to read molecule from.
        :return 'rdkit.Chem.rdchem.Mol': The molecule with properties.
        """
        suppl = Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
        mol = next(suppl)
        Chem.GetSymmSSSR(mol)
        try:
            mono_atom_idxs = mol.GetProp(cls.MONO_ATOM_IDX).split()
        except KeyError:
            return mol
        for atom, mono_atom_idx in zip(mol.GetAtoms(), mono_atom_idxs):
            atom.SetProp(cls.MONO_ATOM_IDX, mono_atom_idx)
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
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
