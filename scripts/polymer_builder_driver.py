# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This polymer driver builds polymers from constitutional repeat units and pack
molecules into condensed phase amorphous cell.

'mpirun -np 4 lmp_mpi -in polymer_builder.in' runs with 4 processors
'lmp_serial -in polymer_builder.in' runs with 1 processor
"""
import os
import sys
import copy
import lammps
import functools
import collections
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from nemd import oplsua
from nemd import symbols
from nemd import jobutils
from nemd import logutils
from nemd import fragments
from nemd import fileutils
from nemd import rdkitutils
from nemd import pnames
from nemd import structutils
from nemd import parserutils
from nemd import environutils

FlAG_CRU = 'cru'
FlAG_CRU_NUM = '-cru_num'
FlAG_MOL_NUM = '-mol_num'
FlAG_DENSITY = '-density'
FlAG_CELL = '-cell'
GRID = 'grid'
PACK = 'pack'
GROW = 'grow'
FLAG_SEED = jobutils.FLAG_SEED

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')


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
    :param timestamp bool: append time information after the message
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_warning(msg):
    """
    Print this warning message into log file.

    :param msg: the msg to print
    """
    if not logger:
        return
    logger.warning(msg)


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


class Validator:

    def __init__(self, options):
        self.options = options

    def run(self):
        """
        Main method to run the validation.
        """
        self.cruNum()
        self.molNum()

    def cruNum(self):
        """
        Validate (or set) the number of repeat units.
        """
        if self.options.cru_num is None:
            self.options.cru_num = [1] * len(self.options.cru)
            return
        if len(self.options.cru_num) == len(self.options.cru):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.cru_num)} cru found.')

    def molNum(self):
        """
        Validate (or set) the number of molecules.
        """
        if self.options.mol_num is None:
            self.options.mol_num = [1] * len(self.options.cru_num)
            return
        if len(self.options.cru_num) == len(self.options.mol_num):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.mol_num)} molecules found.')


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    validator = Validator(options)
    try:
        validator.run()
    except ValueError as err:
        parser.error(err)
    return validator.options


class AmorphousCell(object):
    """
    Build amorphous cell from molecules. (polymers may be built from monomers first)
    """

    MINIMUM_DENSITY = 0.001

    def __init__(self, options, ff=None):
        """
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        :param ff str: the force field filepath
        """
        self.options = options
        self.polymers = []
        self.molecules = {}
        self.mols = {}
        self.ff = ff
        self.box = None
        self.density = None
        if self.ff is None:
            self.ff = oplsua.get_opls_parser()

    def run(self):
        """
        Main method to build the cell.
        """
        self.setPolymers()
        self.setGriddedCell()
        self.setPackedCell()
        self.setGrowedCell()
        self.write()

    def setPolymers(self):
        """
        Build polymer from monomers if provided.
        """
        for cru, cru_num, mol_num in zip(self.options.cru,
                                         self.options.cru_num,
                                         self.options.mol_num):
            polymer = Polymer(cru, cru_num, mol_num, options=self.options)
            polymer.run()
            self.polymers.append(polymer)

    def setGriddedCell(self):
        """
        Build gridded cell.
        """
        if self.options.cell != GRID:
            return
        struct = structutils.GriddedStruct([x.polym for x in self.polymers])
        struct.run()
        self.mols = struct.mols

    def setPackedCell(self, mini_density=MINIMUM_DENSITY):
        """
        Build packed cell.

        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        if self.options.cell != PACK:
            return
        self.createCell(cell_type=PACK, mini_density=mini_density)

    def setGrowedCell(self, mini_density=0.01):
        """
        Build packed cell.

        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        if self.options.cell != GROW:
            return
        self.createCell(cell_type=GROW, mini_density=mini_density)

    def createCell(self, cell_type=PACK, mini_density=MINIMUM_DENSITY):
        """
        Create amorphous cell.

        :param cell_type: the algorithm type for amorphous cell.
        :type cell_type: str
        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        Struct = structutils.PackedStruct if cell_type == PACK else structutils.GrownStruct
        struct = Struct([x.polym for x in self.polymers],
                        ff=self.polymers[0].ff,
                        options=self.options)
        struct.setDataReader()
        density = self.options.density
        mini_density = min([mini_density, density / 5.])
        delta = min([0.1, (density - mini_density) / 4])
        while density >= mini_density:
            try:
                struct.runWithDensity(density)
            except structutils.DensityError:
                density -= delta if density > mini_density else mini_density
                log(f'Density is reduced to {density:.4f} g/cm^3')
            else:
                break

        self.density = density
        self.box = struct.box
        self.mols = struct.mols

    def write(self):
        """
        Write amorphous cell into data file
        """
        lmw = oplsua.LammpsData(self.mols,
                                ff=self.ff,
                                jobname=self.options.jobname,
                                box=self.box,
                                options=self.options)
        lmw.writeData(adjust_coords=False)
        if self.density is None or not np.isclose(lmw.density, self.density):
            log_warning(
                f'The density of the final data file is {lmw.density:.4g} kg/cm^3'
            )
        if round(lmw.total_charge, 4):
            log_warning(
                f'The system has a net charge of {lmw.total_charge:.4f}')
        lmw.writeLammpsIn()
        log(f'Data file written into {lmw.datafile}')
        log(f'In script written into {lmw.lammps_in}')
        jobutils.add_outfile(lmw.datafile, jobname=self.options.jobname)
        jobutils.add_outfile(lmw.lammps_in,
                             jobname=self.options.jobname,
                             set_file=True)


class Polymer(object):
    """
    Class to build a polymer from monomers.
    """

    ATOM_ID = oplsua.LammpsData.ATOM_ID
    TYPE_ID = oplsua.TYPE_ID
    BOND_ATM_ID = oplsua.BOND_ATM_ID
    RES_NUM = oplsua.RES_NUM
    WATER_TIP3P = oplsua.OplsTyper.WATER_TIP3P
    IMPLICIT_H = oplsua.IMPLICIT_H
    MOL_NUM = 'mol_num'
    MONO_ATOM_IDX = 'mono_atom_idx'
    CAP = 'cap'
    HT = 'ht'
    POLYM_HT = pnames.POLYM_HT
    IS_MONO = pnames.IS_MONO
    MONO_ID = pnames.MONO_ID
    CONFORMER_NUM = 'conformer_num'

    def __init__(self, cru, cru_num, mol_num, options=None):
        """
        :param cru str: the smiles string for monomer
        :param cru_num int: the number of monomers per polymer
        :param mol_num int: the number of molecules of this type of polymer
        :param options 'argparse.Namespace': command line options
        """
        self.cru = cru
        self.cru_num = cru_num
        self.mol_num = mol_num
        self.options = options
        self.polym = None
        self.polym_Hs = None
        self.box = None
        self.cru_mol = None
        self.smiles = None
        self.buffer = oplsua.LammpsData.BUFFER
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
        self.setConformers()

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
        wmodel = self.options.force_field.model
        ff_typer = oplsua.OplsTyper(self.polym, wmodel=wmodel)
        ff_typer.run()

    def embedMol(self, trans=False):
        """
        Embed the molecule with coordinates.

        :param trans bool: If True, all_trans conformer without entanglements is
            built.
        """

        if self.polym.GetNumAtoms() <= 200 and not trans:
            with rdkitutils.CaptureLogger() as logs:
                # Mg+2 triggers
                # WARNING UFFTYPER: Warning: hybridization set to SP3 for atom 0
                # ERROR UFFTYPER: Unrecognized charge state for atom: 0
                AllChem.EmbedMolecule(self.polym, useRandomCoords=True)
                [log_debug(f'{x} {y}') for x, y in logs.items()]
            Chem.GetSymmSSSR(self.polym)
            return

        trans_conf = Conformer(self.polym,
                               self.cru_mol,
                               options=self.options,
                               trans=trans)
        trans_conf.run()

    def setConformers(self):
        """
        Set multiple conformers based on the first one.
        """
        self.polym.SetIntProp(self.CONFORMER_NUM, self.mol_num)
        for _ in range(self.polym.GetIntProp(self.CONFORMER_NUM) - 1):
            # copy.copy(self.polym.GetConformer(0)) raise RuntimeError
            # Pickling of "rdkit.Chem.rdchem.Conformer" instances is not enabled
            conf = copy.deepcopy(self.polym).GetConformer(0)
            self.polym.AddConformer(conf, assignId=True)


class Conformer(object):
    """
    Conformer coordinate assignment.
    """

    CAP = Polymer.CAP
    MONO_ATOM_IDX = Polymer.MONO_ATOM_IDX
    MONO_ID = Polymer.MONO_ID
    OUT_EXTN = '.sdf'
    CONFORMER_NUM = Polymer.CONFORMER_NUM

    def __init__(self,
                 polym,
                 original_cru_mol,
                 options=None,
                 trans=True,
                 jobname=None,
                 minimization=True):
        """
        :param polym 'rdkit.Chem.rdchem.Mol': the polymer to set conformer
        :param original_cru_mol 'rdkit.Chem.rdchem.Mol': the monomer mol
            constructing the polymer
        :param options 'argparse.Namespace': command line options.
        :param trans bool: Whether all-tran conformation is requested.
        :param jobname str: The jobname
        :param minimization bool: Whether LAMMPS minimization is performed.
        """
        self.polym = polym
        self.original_cru_mol = original_cru_mol
        self.options = options
        self.trans = trans
        self.jobname = jobname
        self.minimization = minimization
        self.ff = oplsua.get_opls_parser()
        if self.jobname is None:
            self.jobname = 'conf_search'
        self.relax_dir = '_relax'
        self.data_file = 'polym.data'
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
        self.cru_conf = structutils.PackedMol(self.cru_mol).GetConformer(0)

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
        self.cru_bk_aids = nx.shortest_path(graph, *cap_idxs)

    def transAndRotate(self):
        """
        Set trans-conformer with translation and rotation.
        """

        for dihe in zip(self.cru_bk_aids[:-3], self.cru_bk_aids[1:-2],
                        self.cru_bk_aids[2:-1], self.cru_bk_aids[3:]):
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conf, *dihe, 180)

        cntrd = self.cru_conf.centroid(aids=self.cru_bk_aids)
        self.cru_conf.translate(-np.array(cntrd))
        abc_norms = self.getABCVectors()
        abc_targeted = np.eye(3)
        self.cru_conf.rotate(abc_norms, abc_targeted)

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

        bh_xyzs = np.array(
            [self.cru_conf.GetAtomPosition(x) for x in self.cru_bk_aids])
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
        bk_ids = set(self.cru_bk_aids)
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
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conf, *dihe, 90)

    def setXYZAndVect(self):
        """
        Set the XYZ of the non-capping atoms and translational vector, preparing
        for the monomer coordinate shifting.
        """

        cap_ht = [x for x in self.cru_mol.GetAtoms() if x.HasProp(self.CAP)]
        cap_ht = [(x.GetIdx(), x.GetNeighbors()[0].GetIdx()) for x in cap_ht]
        middle_points = np.array([
            np.mean([self.cru_conf.GetAtomPosition(y) for y in x], axis=0)
            for x in cap_ht
        ])

        self.vector = middle_points[1, :] - middle_points[0, :]
        self.xyzs = {
            x.GetIntProp(self.MONO_ATOM_IDX):
            np.array(self.cru_conf.GetAtomPosition(x.GetIdx()))
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
        self.polym.AddConformer(conformer, assignId=True)
        Chem.GetSymmSSSR(self.polym)

    def adjustConformer(self):
        """
        Adjust the conformer coordinates based on the force field.
        """
        mols = {1: self.polym}
        self.lmw = oplsua.LammpsData(mols,
                                     ff=self.ff,
                                     jobname=self.jobname,
                                     options=self.options)
        if self.minimization:
            return
        self.lmw.setOneMolData(adjust_coords=True)

    def minimize(self):
        """
        Run force field minimizer.
        """

        if not self.minimization:
            return

        with fileutils.chdir(self.relax_dir):
            self.lmw.writeData()
            self.lmw.writeLammpsIn()
            import pdb; pdb.set_trace()
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
        fmol = structutils.GrownMol(self.polym, data_file=data_file)
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


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
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
    parser.add_argument(FLAG_SEED,
                        metavar=FLAG_SEED[1:].upper(),
                        type=parserutils.type_random_seed,
                        help='Set random state using this seed.')
    parser.add_argument(
        FlAG_CELL,
        metavar=FlAG_CELL[1:].upper(),
        choices=[GRID, PACK, GROW],
        default=GROW,
        help=f'Amorphous cell type: \'{GRID}\' grids the space and '
        f'put molecules into sub-cells; \'{PACK}\' randomly '
        f'rotates and translates molecules; \'{GROW}\' grows '
        f'molecules from the smallest rigid fragments.')
    parser.add_argument(
        FlAG_DENSITY,
        metavar=FlAG_DENSITY[1:].upper(),
        type=functools.partial(parserutils.type_ranged_float,
                               bottom=0,
                               included_bottom=False,
                               top=30),
        default=0.5,
        help=f'The density used for {PACK} and {GROW} amorphous cell. (g/cm^3)'
    )
    parserutils.add_md_arguments(parser)
    parserutils.add_job_arguments(parser,
                                  jobname=environutils.get_jobname(JOBNAME))
    return parser


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    cell = AmorphousCell(options)
    cell.run()
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
