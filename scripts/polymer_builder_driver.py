# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This polymer driver builds polymers from constitutional repeat units and pack
molecules into condensed phase amorphous cell.
"""
import os
import sys
import math
import copy
import scipy
import lammps
import itertools
import functools
import collections
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

from nemd import traj
from nemd import oplsua
from nemd import symbols
from nemd import jobutils
from nemd import logutils
from nemd import fragments
from nemd import fileutils
from nemd import rdkitutils
from nemd import prop_names
from nemd import structutils
from nemd import parserutils
from nemd import environutils
from nemd import conformerutils

FlAG_CRU = 'cru'
FlAG_CRU_NUM = '-cru_num'
FlAG_MOL_NUM = '-mol_num'
FlAG_DENSITY = '-density'
FlAG_CELL = '-cell'
GRID = 'grid'
PACK = 'pack'
GROW = 'grow'
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


def log_warning(msg, timestamp=False):
    """
    Print this warning message into log file.

    :param msg: the msg to print
    :param timestamp bool: the msg to be printed
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
        self.cruNum()
        self.molNum()

    def cruNum(self):
        if self.options.cru_num is None:
            self.options.cru_num = [1] * len(self.options.cru)
            return
        if len(self.options.cru_num) == len(self.options.cru):
            return
        raise ValueError(f'{len(self.options.cru_num)} cru num defined, but '
                         f'{len(self.options.cru_num)} cru found.')

    def molNum(self):
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
        self.box = None
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
        for cru, cru_num, in zip(self.options.cru, self.options.cru_num):
            polym = Polymer(cru, cru_num, options=self.options)
            polym.run()
            self.polymers.append(polym)

    def setGriddedCell(self):
        """
        Build gridded cell.
        """
        if self.options.cell != GRID:
            return
        cell = GridCell(self.polymers, self.options.mol_num)
        cell.run()
        self.mols = cell.mols

    def setPackedCell(self, mini_density=MINIMUM_DENSITY):
        """
        Build packed cell.
        """
        if self.options.cell != PACK:
            return
        self.createCell(cell_type=PACK, mini_density=mini_density)

    def setGrowedCell(self, mini_density=0.01):
        """
        Build packed cell.
        """
        if self.options.cell != GROW:
            return
        self.createCell(cell_type=GROW, mini_density=mini_density)

    def createCell(self, cell_type=PACK, mini_density=MINIMUM_DENSITY):

        cell_builder = PackedCell if cell_type == PACK else GrowedCell
        cell = cell_builder(self.polymers, self.options)
        cell.setMols()
        cell.setDataReader()
        cell.setAtomMapNum()
        density = self.options.density

        while density >= mini_density:
            try:
                cell.runWithDensity(density)
            except DensityError:
                density -= 0.1 if density > 0.1 else 0.01
                log(f'Density is reduced to {density:.4f} g/cm^3')
            else:
                break

        self.box = cell.box
        self.mols = cell.mols

    def write(self):
        """
        Write amorphous cell into data file
        """
        lmw = oplsua.LammpsData(self.mols,
                                self.ff,
                                self.jobname,
                                box=self.box,
                                options=self.options)
        lmw.writeData(adjust_coords=False)
        if round(lmw.total_charge, 4):
            log_warning(
                f'The system has a net charge of {lmw.total_charge:.4f}')
        lmw.writeLammpsIn()
        log(f'Data file written into {lmw.lammps_data}')
        log(f'In script written into {lmw.lammps_in}')


class GridCell:
    """
    Grid the space and place polymers into the sub-cells.
    """

    def __init__(self, polymers, polym_nums):
        """
        :param polymers 'Polymer': one polymer object for each type
        :param polym_nums list: number of polymers per polymer type
        """
        self.polymers = polymers
        self.polym_nums = polym_nums
        self.mols = {}

    def run(self):
        """
        Create gridded amorphous cell.
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
        for polym, mol_num in zip(self.polymers, self.polym_nums):
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


class PackedCell:
    """
    Pack polymer by random rotation and translation.
    """

    MAX_TRIAL_PER_DENSITY = 100
    MAX_TRIAL_PER_MOL = 10000

    def __init__(self, polymers, options):
        """
        :param polymers 'Polymer': one polymer object for each type
        :param options 'argparse.Namespace': command line options
        """
        self.polymers = polymers
        self.options = options
        self.box = None
        self.mols = {}

    def run(self):
        """
        Create amorphous cell by randomly placing molecules with random
        orientations.
        """

        self.setBoxes()
        self.setMols()
        self.setDataReader()
        self.setAtomMapNum()
        self.setFrameAndDcell()
        self.placeMols()

    def runWithDensity(self, density):
        """
        Create amorphous cell of the target density by randomly placing
        molecules with random orientations.

        NOTE: the final density of the output cell may be smaller than the
        target if the max number of trial attempt is reached.

        :param density float: the target density
        """
        self.density = density
        self.setBoxes()
        self.setFrameAndDcell()
        self.placeMols()

    def setBoxes(self):
        """
        Set periodic boundary box size.
        """
        weight = sum(x.mw * y
                     for x, y in zip(self.polymers, self.options.mol_num))
        vol = weight / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = [0, edge, 0, edge, 0, edge]
        log(f'Cubic box of size {edge:.2f} angstrom is created.')

    def setMols(self):
        """
        Set molecules.
        """
        mols = [
            copy.copy(x.polym)
            for x, y in zip(self.polymers, self.options.mol_num)
            for _ in range(y)
        ]
        self.mols = {i: x for i, x in enumerate(mols, start=1)}
        for mol_id, mol in self.mols.items():
            mol.SetIntProp(prop_names.MOL_ID, mol_id)

    def setDataReader(self):
        """
        Set data reader with clash parameters.
        """

        lmw = oplsua.LammpsData(self.mols,
                                self.polymers[0].ff,
                                'tmp',
                                options=self.options)
        lmw.writeData()
        self.df_reader = oplsua.DataFileReader('tmp.data')
        self.df_reader.run()
        self.df_reader.setClashParams()

    def setAtomMapNum(self):
        """
        Set atom force field id.
        """
        for mol_id in self.mols.keys():
            mol = self.mols[mol_id]
            atom_fids = self.df_reader.mols[mol_id]
            for atom, atom_fid in zip(mol.GetAtoms(), atom_fids):
                atom.SetAtomMapNum(atom_fid)

    def setFrameAndDcell(self):
        """
        Set the trajectory frame and distance cell.
        """
        index = [atom.id for atom in self.df_reader.atoms.values()]
        xyz = [atom.xyz for atom in self.df_reader.atoms.values()]
        self.frm = traj.Frame(xyz=xyz, index=index, box=self.box)
        self.dcell = traj.DistanceCell(self.frm)
        self.dcell.setUp()

    def placeMols(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """
        trial_num = 1
        while trial_num <= max_trial:
            self.extg_aids = set()
            for mol_id in self.df_reader.mols.keys():
                try:
                    self.placeMol(mol_id)
                except MolError:
                    log_debug(f'{trial_num} trail fails. '
                              f'(Only {mol_id - 1} / {len(self.mols)} '
                              f'molecules placed in the cell.)')
                    trial_num += 1
                    break
            else:
                # All molecules successfully placed (no break)
                return
        raise DensityError

    def placeMol(self, mol_id, max_trial=MAX_TRIAL_PER_MOL):
        """
        Place molecules one molecule into the cell without clash.

        :param mol_id int: the molecule id of the molecule to be placed into the
            cell.
        :param max_trial int: the max trial number for each molecule to be placed
            into the cell.
        """
        aids = self.df_reader.mols[mol_id]
        trial_per_mol = 1
        while trial_per_mol <= max_trial:
            self.translateMol(mol_id)
            if not self.hasClashes(aids):
                self.extg_aids.update(aids)
                # Only update the distance cell after one molecule successful
                # placed into the cell as only inter-molecular clashes are
                # checked for packed cell.
                self.dcell.setUp()
                return
            trial_per_mol += 1
        if trial_per_mol > max_trial:
            raise MolError

    def translateMol(self, mol_id, aids=None):
        """
        Do translation and rotation to the molecule so that the centroid will be
        randomly point in the cell and the orientation is also randomly picked.

        :param mol_id int: the molecule id of the molecule to be placed into the
            cell.
        :param aids list: list of atom ids whose centroid is translated.
        """
        mol = self.mols[mol_id]
        conf = mol.GetConformer()
        centroid = np.array(conformerutils.centroid(conf, aids=aids))
        conformerutils.translation(conf, -centroid)
        conformerutils.rand_rotate(conf)
        conformerutils.translation(conf, self.frm.getPoint())
        aids = [x.GetAtomMapNum() for x in mol.GetAtoms()]
        self.frm.loc[aids] = conf.GetPositions()

    def hasClashes(self, aids):
        """
        Whether these atoms have any clashes with the existing atoms in the cell.

        :param aids list of int: the atom ids to check clashes
        """
        for id, row in self.frm.loc[aids].iterrows():
            clashes = self.dcell.getClashes(row,
                                            included=self.extg_aids,
                                            radii=self.df_reader.radii,
                                            excluded=self.df_reader.excluded)
            if clashes:
                return True
        return False


class MolError(RuntimeError):
    """
    When max number of the failure for this molecule has been reached.
    """
    pass


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class GrowedCell(PackedCell):
    """
    Grow the polymers from bit to full.
    """

    MAX_TRIAL_PER_DENSITY = 10
    MAX_TRIAL_PER_MOL = 10

    def __init__(self, *arg, **kwarg):
        """
        :param polymers 'Polymer': one polymer object for each type
        :param polym_nums list: number of polymers per polymer type
        """
        super().__init__(*arg, **kwarg)

    def placeMols(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """

        frag_mols = fragments.FragMols(self.mols,
                                       data_file='tmp.data',
                                       box=self.box,
                                       logger=logger)
        frag_mols.run()


class Polymer(object):
    """
    Class to build a polymer from monomers.
    """

    ATOM_ID = oplsua.LammpsData.ATOM_ID
    TYPE_ID = oplsua.TYPE_ID
    BOND_ATM_ID = oplsua.BOND_ATM_ID
    RES_NUM = oplsua.RES_NUM
    NEIGHBOR_CHARGE = oplsua.LammpsData.NEIGHBOR_CHARGE
    WATER_TIP3P = oplsua.OplsTyper.WATER_TIP3P
    IMPLICIT_H = oplsua.IMPLICIT_H
    MOL_NUM = 'mol_num'
    MONO_ATOM_IDX = 'mono_atom_idx'
    CAP = 'cap'
    HT = 'ht'
    POLYM_HT = prop_names.POLYM_HT
    IS_MONO = prop_names.IS_MONO
    MONO_ID = prop_names.MONO_ID

    def __init__(self, cru, cru_num, options=None):
        """
        :param cru str: the smiles string for monomer
        :param cru_num int: the number of monomers per polymer
        :param options 'argparse.Namespace': command line options
        """
        self.cru = cru
        self.cru_num = cru_num
        self.options = options
        self.polym = None
        self.polym_Hs = None
        self.box = None
        self.cru_mol = None
        self.molecules = []
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

    @property
    def molecular_weight(self):
        atypes = [x.GetIntProp(self.TYPE_ID) for x in self.polym.GetAtoms()]
        return sum(self.ff.atoms[x].mass for x in atypes)

    mw = molecular_weight


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
                 options=None,
                 trans=True,
                 jobname=None,
                 minimization=False):
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
        self.cru_bk_aids = nx.shortest_path(graph, *cap_idxs)

    def transAndRotate(self):
        """
        Set trans-conformer with translation and rotation.
        """

        for dihe in zip(self.cru_bk_aids[:-3], self.cru_bk_aids[1:-2],
                        self.cru_bk_aids[2:-1], self.cru_bk_aids[3:]):
            Chem.rdMolTransforms.SetDihedralDeg(self.cru_conformer, *dihe, 180)

        cntrd = conformerutils.centroid(self.cru_conformer,
                                        aids=self.cru_bk_aids)

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

        bh_xyzs = np.array(
            [self.cru_conformer.GetAtomPosition(x) for x in self.cru_bk_aids])
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
        self.lmw = oplsua.LammpsData(mols,
                                     self.ff,
                                     self.jobname,
                                     options=self.options)
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


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
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
    parser.add_argument(FlAG_SEED,
                        metavar=FlAG_SEED[1:].upper(),
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
                               bottom=AmorphousCell.MINIMUM_DENSITY,
                               top=30),
        default=0.5,
        help=f'The density used for {PACK} and {GROW} amorphous cell. (g/cm^3)'
    )
    jobutils.add_md_arguments(parser)
    jobutils.add_job_arguments(parser)
    return parser


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    cell = AmorphousCell(options, jobname)
    cell.run()
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
