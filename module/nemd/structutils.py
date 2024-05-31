# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles molecular topology and structural editing.
"""
import math
import rdkit
import scipy
import warnings
import itertools
import functools
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs
from scipy.spatial.transform import Rotation

from nemd import traj
from nemd import oplsua
from nemd import pnames
from nemd import logutils

EDGES = 'edges'
WEIGHT = 'weight'

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


class Conformer(rdkit.Chem.rdchem.Conformer):
    """
    A subclass of rdkit.Chem.rdchem.Conformer with additional attributes and methods.
    """

    def __init__(self, *args, mol=None, **kwargs):
        """
        :param mol `rdkit.Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        super().__init__(*args, **kwargs)
        self.mol = mol

    def GetOwningMol(self):
        """
        Get the Mol that owns this conformer.

        :return `rdkit.Chem.rdchem.Mol`: the molecule this conformer belongs to.
        """
        return self.mol

    def SetOwningMol(self, mol):
        """
        Set the Mol that owns this conformer.
        """
        self.mol = mol

    def HasOwningMol(self):
        """
        Returns whether or not this conformer belongs to a molecule.

        :return `bool`: the molecule this conformer belongs to.
        """
        return bool(self.mol)

    def centroid(self, aids=None):
        """
        Compute the centroid of the whole conformer ar the selected atoms.

        :param atom_ids list: the selected atom ids
        """
        if aids is None:
            return Chem.rdMolTransforms.ComputeCentroid(self)

        bv = DataStructs.ExplicitBitVect(self.GetNumAtoms())
        bv.SetBitsFromList(aids)
        weights = rdBase._vectd()
        weights.extend(bv.ToList())
        return Chem.rdMolTransforms.ComputeCentroid(self,
                                                    weights=weights,
                                                    ignoreHs=False)

    def translate(self, vect):
        """
        Do translation on this conformer using this vector.

        :param vect 'numpy.ndarray': translational vector
        """
        mtrx = np.identity(4)
        mtrx[:-1, 3] = vect
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def rotateRandomly(self):
        """
        Randomly rotate the conformer.

        NOTE: the random state is set according to the numpy random seed.
        """
        mtrx = np.identity(4)
        seed = np.random.randint(0, 2**32 - 1)
        mtrx[:-1, :-1] = Rotation.random(random_state=seed).as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def rotate(self, ivect, tvect):
        """
        Rotate the conformer by three initial vectors and three target vectors.

        :param ivect 3x3 'numpy.ndarray': Each row is one initial vector
        :param tvect 3x3 'numpy.ndarray': Each row is one corresponding target vector
        """
        rotation, rmsd = Rotation.align_vectors(tvect, ivect)
        mtrx = np.identity(4)
        mtrx[:-1, :-1] = rotation.as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)


class ConfError(RuntimeError):
    """
    When max number of the failure for this conformer has been reached.
    """
    pass


class PackedConf(Conformer):

    MAX_TRIAL_PER_CONF = 10000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_map = None
        self.frm = None
        self.dcell = None
        self.extg_gids = None
        self.radii = None
        self.excluded = None

    def setReferences(self):
        """
        Set the references to the conformer.
        """
        atoms, gids = self.mol.GetAtoms(), self.mol.id_map[self.GetId()]
        self.id_map = {x.GetIdx(): y for x, y in zip(atoms, gids)}
        self.radii = self.mol.radii
        self.excluded = self.mol.excluded
        self.frm = self.mol.frm
        self.dcell = self.mol.dcell
        self.extg_gids = self.mol.extg_gids

    @property
    @functools.lru_cache(maxsize=None)
    def gids(self):
        """
        Return the global atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return list(self.id_map.values())

    def setConformer(self, max_trial=MAX_TRIAL_PER_CONF):
        """
        Place molecules one molecule into the cell without clash.

        :param max_trial int: the max trial number for each molecule to be placed
            into the cell.
        :raise MolError: if the conformer always has clashes with the existing
            atoms in the cell after the maximum trial.
        """
        for _ in range(max_trial):
            self.translate(-np.array(self.centroid()))
            self.rotateRandomly()
            self.translate(self.frm.getPoint())
            self.frm.loc[self.gids] = self.GetPositions()
            if self.hasClashes():
                continue
            self.extg_gids.update(self.gids)
            # Only update the distance cell after one molecule successful
            # placed into the cell as only inter-molecular clashes are
            # checked for packed cell.
            self.dcell.setUp()
            return
        raise ConfError

    def hasClashes(self):
        """
        Whether the conformer has any clashes with the existing atoms in the cell.

        :param bool: the conformer has clashes or not.
        """
        for id, row in self.frm.loc[self.gids].iterrows():
            clashes = self.dcell.getClashes(row,
                                            included=self.extg_gids,
                                            radii=self.radii,
                                            excluded=self.excluded)
            if clashes:
                return True
        return False


class GrownConf(PackedConf):

    def __init__(self, *args, **kwargs):
        super(GrownConf, self).__init__(*args, **kwargs)
        self.ifrag = None
        self.init_aids = None
        self.init_gids = None
        self.failed_num = 0
        self.frags = []

    def setPositions(self, xyz):
        """
        Reset the positions of the atoms to the original xyz coordinates.

        :return xyz np.ndarray: the xyz coordinates of the molecule.
        """
        for id in range(xyz.shape[0]):
            self.SetAtomPosition(id, xyz[id, :])

    def setReferences(self):
        """
        Pass aid-to-gid map, radii, excluded atoms, and distance cell from the
        molecule object to this conformer.
        """
        super().setReferences()
        self.init_tf = self.mol.init_tf

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments if not, copy to
        current conformer, and set up the fragment objects.
        """
        self.GetOwningMol().fragmentize()
        mol = self.GetOwningMol()
        self.init_aids = mol.init_aids.copy()
        self.init_gids = [self.id_map[x] for x in self.init_aids]
        self.ifrag = mol.ifrag.copyInit(self)
        self.frags = [self.ifrag]

    def getSwingAtoms(self, *dihe):
        """
        Get the swing atoms when the dihedral angle changes.

        :param dihe list of four ints: the atom ids that form a dihedral angle
        :return list of ints: the swing atom ids when the dihedral angle changes.
        """
        oxyz = self.GetPositions()
        oval = self.getDihedralDeg(dihe)
        self.setDihedralDeg(dihe, oval + 5)
        xyz = self.GetPositions()
        changed = np.isclose(oxyz, xyz)
        self.setDihedralDeg(dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def setDihedralDeg(self, dihe, val):
        """
        Set angle degree of the given dihedral.

        :param dihe tuple of int: the dihedral atom indices.
        :param val float: the angle degree.
        """
        Chem.rdMolTransforms.SetDihedralDeg(self, *dihe, val)

    def getDihedralDeg(self, dihe):
        """
        Get the angle degree of the given dihedral.

        :param dihe tuple of int: the dihedral atom indices.
        :param return float: the angle degree.
        """
        return Chem.rdMolTransforms.GetDihedralDeg(self, *dihe)

    @functools.lru_cache(maxsize=None)
    def getNumFrags(self):
        """
        Return the number of the total fragments.

        :return int: number of the total fragments.
        """
        # ifrag without dihe means rigid body and counts as 1 fragment
        return len(self.ifrag.fragments()) + 1 if self.ifrag.dihe else 1

    def placeInitFrag(self):
        """
        Place the initiator fragment into the cell with random position, random
        orientation, and large separation.

        :param frag 'fragments.Fragment': the fragment to place

        :raise ValueError: when no void to place the initiator fragment of the
            dead molecule.
        """

        self.dcell.rmClashNodes()
        points = self.dcell.getVoids()
        for point in points:
            centroid = np.array(self.centroid(aids=self.init_aids))
            self.translate(-centroid)
            self.rotateRandomly()
            self.translate(point)
            self.frm.loc[self.gids] = self.GetPositions()

            if self.hasClashes(self.init_gids):
                continue
            # Only update the distance cell after one molecule successful
            # placed into the cell as only inter-molecular clashes are
            # checked for packed cell.
            self.add(list(self.init_gids))
            self.init_tf.loc[self.GetId()] = point
            return

        # with open('placeInitFrag.xyz', 'w') as out_fh:
        #     self.frm.write(out_fh,
        #                    dreader=self.df_reader,
        #                    visible=list(self.dcell.extg_gids),
        #                    points=points)
        raise ValueError(f'Failed to relocate the dead molecule. '
                         f'({len(self.dcell.extg_gids)}/{len(self.mols)})')

    def add(self, gids):
        """
        Update trajectory frame, add atoms to the atom cell and existing record.

        :param gids list: gids of the atoms to be added
        """
        self.updateFrm()
        self.dcell.atomCellUpdate(gids)
        self.dcell.addGids(gids)

    def updateFrm(self):
        """
        Update the coordinate frame based on the current conformer.
        """
        self.frm.loc[self.gids] = self.GetPositions()

    def remove(self, gids):
        """
        Remove atoms from the atom cell and existing record.

        :param gids list: gids of the atoms to be removed
        """
        self.dcell.atomCellRemove(gids)
        self.dcell.removeGids(gids)

    def hasClashes(self, gids):
        """
        Whether the atoms has clashes with existing atoms in the cell.

        :param gids list: golabal atom ids to check clashes against.
        :return bool: True if clashes are found.
        """
        for row in [self.frm.loc[x] for x in gids]:
            clashes = self.dcell.getClashes(row,
                                            included=self.dcell.extg_gids,
                                            radii=self.radii,
                                            excluded=self.excluded)
            if clashes:
                return True
        return False

    def setFrag(self):
        """
        Set part of the conformer by rotating the dihedral angle, back moving,
        and relocation.
        """

        frag = self.frags.pop(0)

        if not frag.dihe:
            # ifrag without dihe means rigid body
            return

        if self.setDihedral(frag):
            return

        if self.backMove(frag):
            return

        # The molecule has grown to a dead end
        self.failed_num += 1
        self.ifrag.resetVals()
        # The method backmove() deletes some extg_gids
        self.dcell.resetGraph()
        self.placeInitFrag()
        self.reportRelocation()
        log_debug(f'{len(self.dcell.extg_gids)} atoms placed.')

    def setDihedral(self, frag):
        """
        Set part of the conformer by rotating the dihedral angle.

        :param frag 'fragments.Fragment': fragment to set the dihedral angle.
        :return bool: True if successfully place one fragment.
        """
        while frag.vals:
            frag.setDihedralDeg()
            self.updateFrm()
            if self.hasClashes(frag.gids):
                continue
            # Successfully grew one fragment
            self.frags += frag.nfrags
            self.add(frag.gids)
            # self.reportStatus()
            return True

        return False

    def backMove(self, frag):
        """
        Back move fragment so that the obstacle can be walked around later.

        :param frag 'fragments.Fragment': fragment to perform back move
        :return bool: True if back move is successful.
        """
        # 1）Find the previous fragment with available dihedral candidates.
        pfrag = frag.getPreAvailFrag()
        found = bool(pfrag)
        frag = pfrag if found else self.ifrag
        # 2）Find the next fragments who have been placed into the cell.
        nxt_frags = frag.getNxtFrags()
        [x.resetVals() for x in nxt_frags]
        ratom_gids = [y for x in nxt_frags for y in x.gids]
        if not found:
            ratom_gids += frag.conf.init_gids
        self.remove(ratom_gids)
        # 3）Fragment after the next fragments were added to the growing
        # frags before this backmove step.
        nnxt_frags = [y for x in nxt_frags for y in x.nfrags]
        self.frags = [frag] + [x for x in self.frags if x not in nnxt_frags]
        log_debug(f"{len(self.dcell.extg_gids)}, {len(frag.vals)}: {frag}")
        return found

    def reportRelocation(self):
        """
        Report the status after relocate an initiator fragment.

        :param frag 'fragments.Fragment': the relocated fragment
        """

        idists = self.init_tf.pairDists()
        dists = self.dcell.getDistsWithIds(self.init_gids)
        log_debug(f"Relocate the initiator of {self.GetId()} conformer "
                 f"(initiator: {idists.min():.2f}-{idists.max():.2f}; "
                 f"close contact: {dists.min():.2f}) ")


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, ff=None, **kwargs):
        """
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.conf_id = 0
        self.confs = None

    def setConformerId(self, conf_id):
        """
        Set the selected conformer id.

        :param conf_id int: the conformer id to select.
        """
        self.conf_id = conf_id

    def GetConformer(self, conf_id=None):
        """
        Get the conformer of the molecule.

        :param conf_id int: the conformer id to get.
        :return `rdkit.Chem.rdchem.Conformer`: the selected conformer.
        """
        if conf_id is None:
            conf_id = self.conf_id
        if self.confs is None:
            self.initConformers()
        return self.confs[conf_id]

    def GetConformers(self):
        """
        Get the conformers of the molecule.

        :return list of conformers: the conformers of the molecule.
        """
        if self.confs is not None:
            return [x for x in self.confs.values()]
        self.initConformers()
        return [x for x in self.confs.values()]

    def initConformers(self, ConfClass=Conformer):
        """
        Set the conformers of the molecule.

        :param ConfClass (sub)class 'rdkit.Chem.rdchem.Conformer': the conformer
            class to instantiate conformers.
        """
        confs = super().GetConformers()
        self.confs = {i: ConfClass(x, mol=self) for i, x in enumerate(confs)}

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight


class GriddedMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol to handle gridded conformers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # size = xyz span + buffer
        self.buffer = np.array([4, 4, 4])
        # The number of molecules per box edge
        self.mol_num = np.array([1, 1, 1])
        # The xyz shift within one box
        self.vecs = []

    @property
    def size(self):
        """
        Return the box size of this molecule.
        """
        # Grid layout assumes all conformers from one molecule are the same
        xyzs = self.GetConformer().GetPositions()
        return (xyzs.max(axis=0) - xyzs.min(axis=0)) + self.buffer

    @property
    def box_num(self):
        """
        Return the number of boxes (the largest molecule size) needed to place
            all conformers.
        """
        return math.ceil(self.GetNumConformers() / np.prod(self.mol_num))

    def setConfNumPerEdge(self, size):
        """
        Set the number of molecules per edge of the box.

        :param size np.ndarray: the box size (the largest molecule size) to
            place this conformer in.
        """
        self.mol_num = np.floor(size / self.size).astype(int)

    def setVecs(self, size):
        """
        Set the translational vectors for this conformer so that this conformer
        can be placed in the given box (the largest molecule size).

        :param size np.ndarray: the box size to place this molecule in.
        """
        ptc = [np.linspace(-0.5, 0.5, x, endpoint=False) for x in self.mol_num]
        ptc = [x - x.mean() for x in ptc]
        self.vecs = [
            x * size for x in itertools.product(*[[y for y in x] for x in ptc])
        ]

    def setConformers(self, vector):
        """
        Set the conformers of this molecule based on the shifting vector.

        :param vector np.ndarray: the translational vector to move the conformer
            by multiple boxes distances.
        """
        mol_num_per_box = np.prod(self.mol_num)
        for idx in range(min([self.GetNumConformers(), mol_num_per_box])):
            vecs = vector + self.vecs[idx]
            self.GetConformer().translate(vecs)
            self.conf_id += 1


class PackedMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, ff=None, **kwargs):
        """
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.id_map = None
        self.radii = None
        self.excluded = None
        self.frm = None
        self.dcell = None
        self.extg_gids = None

    def initConformers(self, ConfClass=PackedConf):
        """
        See parant class for details.
        """
        return super().initConformers(ConfClass=ConfClass)


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """

    def __init__(self, mols, MolClass=Mol, ff=None, **kwargs):
        """
        :param mols list of rdkit.Chem.rdchem.Mol: the molecules to be handled.
        :param MolClass subclass of 'rdkit.Chem.rdchem.Mol': the customized
            molecule class
        :param ff 'OplsParser': the force field class.
        """
        self.mols = {
            i: MolClass(x, ff=ff)
            for i, x in enumerate(mols, start=1)
        }
        for mol_id, conf in enumerate(self.conformers, start=1):
            conf.SetId(mol_id)
        self.density = None

    @property
    def conformers(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all
            molecules.
        """
        return [x for y in self.molecules for x in y.GetConformers()]

    @property
    def molecules(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all
            molecules.
        """
        return [x for x in self.mols.values()]


class GriddedStruct(Struct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    def __init__(self, *args, MolClass=GriddedMol, **kwargs):
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.size = np.zeros([3])

    def run(self):
        """
        Set conformers for all molecules.
        """
        self.setSize()
        self.setVectors()
        self.setBox()
        self.setConformers()
        self.setDensity()

    def setSize(self):
        """
        Set the size as the maximum size over all molecules.
        """
        self.size = np.array([x.size for x in self.molecules]).max(axis=0)

    def setVectors(self):
        """
        Set translational vectors based on the box for all molecules.
        """
        for mol in self.molecules:
            mol.setConfNumPerEdge(self.size)
            mol.setVecs(self.size)

    def setBox(self):
        """
        Set the periodic boundary box size.
        """
        # vectors shifts molecules by the largest box size
        total_box_num = sum(x.box_num for x in self.molecules)
        edges = self.size * math.ceil(math.pow(total_box_num, 1. / 3))
        self.box = [0, edges[0], 0, edges[1], 0, edges[2]]
        log_debug(f'Cubic box of size {self.box[0]:.2f} angstrom is created.')

    def setConformers(self):
        """
        Set coordinates.
        """
        idxes = [list(range(x)) for x in map(int, self.box[1::2] / self.size)]
        # vectors shifts molecules by the largest box size
        self.vectors = [x * self.size for x in itertools.product(*idxes)]
        # boxes are filled in random order with all molecules in random order
        molecules = self.molecules[:]
        while molecules:
            mol = np.random.choice(molecules)
            np.random.shuffle(self.vectors)
            vector = self.vectors.pop()
            mol.setConformers(vector)
            if mol.conf_id == mol.GetNumConformers():
                molecules.remove(mol)

    def setDensity(self):
        """
        Set the density of the structure.
        """
        weight = sum(x.mw * x.GetNumConformers() for x in self.molecules)
        vol = np.prod(self.box[1::2])
        vol *= math.pow(scipy.constants.centi / scipy.constants.angstrom, 3)
        self.density = weight * scipy.constants.Avogadro / vol


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class PackedStruct(Struct):
    """
    Pack molecules by random rotation and translation.
    """

    MAX_TRIAL_PER_DENSITY = 100

    def __init__(self,
                 *args,
                 MolClass=PackedMol,
                 ff=None,
                 options=None,
                 **kwargs):
        """
        :param polymers 'Polymer': one polymer object for each type
        :param ff 'OplsParser': the force field class.
        :param options 'argparse.Namespace': command line options
        """
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        MolClass = functools.partial(MolClass, ff=ff)
        super().__init__(*args, MolClass=MolClass, ff=ff, **kwargs)
        self.ff = ff
        self.options = options
        self.df_reader = None
        self.extg_gids = set()

    def run(self):
        """
        Create amorphous cell by randomly placing molecules with random
        orientations.
        """
        self.setBox()
        self.setDataReader()
        self.setFrameAndDcell()
        self.setReferences()
        self.fragmentize()
        self.setConformers()

    def runWithDensity(self, density):
        """
        Create amorphous cell of the target density by randomly placing
        molecules with random orientations.

        NOTE: the final density of the output cell may be smaller than the
        target if the max number of trial attempt is reached.

        :param density float: the target density
        """
        self.density = density
        self.run()

    def fragmentize(self):
        ...

    def setBox(self):
        """
        Set periodic boundary box size.
        """
        weight = sum(x.mw * x.GetNumConformers() for x in self.mols.values())
        vol = weight / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = [0, edge, 0, edge, 0, edge]
        log_debug(f'Cubic box of size {edge:.2f} angstrom is created.')

    def setDataReader(self):
        """
        Set data reader with clash parameters.
        """
        if self.df_reader is not None:
            return
        lmw = oplsua.LammpsData(self.mols, ff=self.ff, options=self.options)
        contents = lmw.writeData(nofile=True)
        self.df_reader = oplsua.DataFileReader(contents=contents)
        self.df_reader.run()
        self.df_reader.setClashParams()

    def setFrameAndDcell(self):
        """
        Set the trajectory frame and distance cell.
        """
        index = [atom.id for atom in self.df_reader.atoms.values()]
        xyz = [atom.xyz for atom in self.df_reader.atoms.values()]
        self.frm = traj.Frame(xyz=xyz, index=index, box=self.box)
        self.dcell = traj.DistanceCell(self.frm)
        self.dcell.setUp()

    def setReferences(self):
        """
        Set references to all molecular and conformers.
        """

        for mol in self.mols.values():
            mol.id_map = self.df_reader.mols
            mol.radii = self.df_reader.radii
            mol.excluded = self.df_reader.excluded
            mol.frm = self.frm
            mol.dcell = self.dcell
            mol.extg_gids = self.extg_gids
            for conf in mol.GetConformers():
                conf.setReferences()

    def setConformers(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """
        trial_id, mol_num = 1, len(self.conformers)
        tenth, threshold, = mol_num / 10., 0
        for trial_id in range(1, max_trial + 1):
            self.extg_gids.clear()
            for conf_id, conf in enumerate(self.conformers):
                try:
                    conf.setConformer()
                except ConfError:
                    log_debug(f'{trial_id} trail fails. '
                              f'(Only {conf_id} / {mol_num} '
                              f'molecules placed in the cell.)')
                    break

                if conf_id < threshold:
                    continue
                conf_num = conf_id + 1
                new_line = "" if conf_id == mol_num else ", [!n]"
                log_debug(f"{int(conf_num / mol_num * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)

            # All molecules successfully placed (no break)
            return
        raise DensityError


class GrownMol(PackedMol):

    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    PATT = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
    IS_MONO = pnames.IS_MONO
    MONO_ID = pnames.MONO_ID
    POLYM_HT = pnames.POLYM_HT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ifrag = None
        self.init_aids = None
        self.setGraph()
        self.rotatable_bonds = self.GetSubstructMatches(self.PATT,
                                                        maxMatches=1000000)

    def initConformers(self, ConfClass=GrownConf):
        """
        See parant class for details.
        """
        return super().initConformers(ConfClass=ConfClass)

    def setGraph(self):
        """
        Get the networkx graph on the molecule.
        """
        self.graph = nx.Graph()
        edges = [(
            x.GetBeginAtom().GetIdx(),
            x.GetEndAtom().GetIdx(),
        ) for x in self.GetBonds()]
        if not edges:
            # When bonds don't exist, just add the atom.
            for atom in self.GetAtoms():
                self.graph.add_node(atom.GetIdx())
            return
        # When bonds exist, add edges and the associated atoms, assuming atoms in
        # one molecule are bonded.
        self.graph.add_edges_from(edges)
        for edge in edges:
            for idx in range(2):
                node = self.graph.nodes[edge[idx]]
                try:
                    node[EDGES].append(edge)
                except KeyError:
                    node[EDGES] = [edge]

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        if self.ifrag is not None:
            return
        # dihe is not known and will be handled in setFragments()
        self.ifrag = Fragment([], self.GetConformer(), delay=True)
        self.ifrag.setFragments()
        frags = self.ifrag.fragments()
        frag_aids_set = set([y for x in frags for y in x.aids])
        all_aids = set([x.GetIdx() for x in self.GetAtoms()])
        self.init_aids = list(all_aids.difference(frag_aids_set))

    def findHeadTailPair(self):
        """
        If the molecule is built from monomers, the atom pairs from
        selected from the first and last monomers.

        :return list or iterator of int tuple: each tuple is an atom id pair
        """
        if not self.HasProp(self.IS_MONO) or not self.GetBoolProp(
                self.IS_MONO):
            return [(None, None)]

        ht_mono_ids = {
            x.GetProp(self.MONO_ID): []
            for x in self.GetAtoms() if x.HasProp(self.POLYM_HT)
        }
        for atom in self.GetAtoms():
            try:
                ht_mono_ids[atom.GetProp(self.MONO_ID)].append(atom.GetIdx())
            except KeyError:
                pass

        st_atoms = list(ht_mono_ids.values())
        sources = st_atoms[0]
        targets = [y for x in st_atoms[1:] for y in x]

        return itertools.product(sources, targets)

    def findPath(self, source=None, target=None):
        """
        Find the shortest path between source and target. If source and target
        are not provided, shortest paths between all pairs are computed and the
        long path is returned.

        :param source int: the atom id that serves as the source.
        :param target int: the atom id that serves as the target.
        :return list of ints: the atom ids that form the shortest path.
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shortest_path = nx.shortest_path(self.graph,
                                             source=source,
                                             target=target)

        if target is not None:
            shortest_path = {target: shortest_path}
        if source is not None:
            shortest_path = {source: shortest_path}
        path_length, path = -1, None
        for a_source_node, target_path in shortest_path.items():
            for a_target_node, a_path in target_path.items():
                if path_length >= len(a_path):
                    continue
                source_node = a_source_node
                target_node = a_target_node
                path = a_path
                path_length = len(a_path)
        return source_node, target_node, path

    def isRotatable(self, bond):
        """
        Whether the bond between the two atoms is rotatable.

        :param bond list or tuple of two ints: the atom ids of two bonded atoms
        :return bool: Whether the bond is rotatable.
        """

        in_ring = self.GetBondBetweenAtoms(*bond).IsInRing()
        single = tuple(sorted(bond)) in self.rotatable_bonds
        return not in_ring and single

    def copy(self, conf):
        """
        Copy the current GrownMol object and set the new conformer.
        NOTE: dihedral value candidates, existing atom ids, and fragment references
        are copied. Other attributes such as the graph, rotatable bonds, and frames
        are just referred to the original object.

        :param conf 'rdkit.Chem.rdchem.Conformer': the new conformer.
        """
        mol = GrownMol(self)
        mol.conf = conf
        mol.ifrag = self.ifrag.copyInit(mol=mol) if self.ifrag else None
        mol.extg_aids = self.extg_aids.copy()
        mol.frm = self.frm
        mol.graph = self.graph
        mol.rotatable_bonds = self.rotatable_bonds
        return mol

    @property
    def molecule_id(self):
        """
        Return molecule id

        :return int: the molecule id.
        """
        return self.conf.GetId()


class GrownStruct(PackedStruct):

    MAX_TRIAL_PER_DENSITY = 10

    def __init__(self, *args, MolClass=GrownMol, **kwargs):
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.failed_num = 0  # The failed attempts in growing molecules
        self.mol_num = None  # the last reported growing molecule number
        self.init_tf = None

    def setFrameAndDcell(self):
        """
        Set distance cell parameters.
        """

        # memory saving flaot16 to regular float32
        # Using [0][1][2] as the cell, atoms in [0] and [2], are at least
        # Separated by 1 max_clash_dist, meaning no clashes.
        self.cell_cut = float(self.df_reader.radii.max())
        super().setFrameAndDcell()

    def setReferences(self):
        for mol in self.mols.values():
            mol.init_tf = self.init_tf
        super().setReferences()

    def fragmentize(self):
        """
        Break the molecule into the smallest rigid fragments.
        """
        if all([x.ifrag for x in self.conformers]):
            return

        for conf in self.conformers:
            conf.fragmentize()
        total_frag_num = sum([x.getNumFrags() for x in self.conformers])
        log_debug(f"{total_frag_num} fragments in total.")

    def setConformers(self):
        """
        Set conformer coordinates without clashes.
        """
        self.setXYZ()
        self.setInitFrm()
        self.setDcell()
        self.placeInitFrags()
        conformers = [x for x in self.conformers]
        while conformers:
            conf = conformers.pop(0)
            if not conf.frags:
                continue
            conf.setFrag()
            conformers.append(conf)

    def setXYZ(self):
        for conf in self.conformers:
            xyz = self.df_reader.getMolXYZ(conf.GetId())
            conf.setPositions(xyz)

    def setInitFrm(self):
        """
        Set the traj frame for initiators.
        """
        data = np.full((len(self.conformers), 3), np.inf)
        index = [x.GetId() for x in self.conformers]
        self.init_tf = traj.Frame(xyz=data, index=index, box=self.box)

    def setDcell(self):
        """
        Set distance cell for neighbor atom and graph for voids searching.
        """
        self.updateFrm()
        self.dcell = traj.DistanceCell(frm=self.frm, cut=self.cell_cut)
        self.dcell.setUp()
        self.dcell.setGraph(len(self.conformers))
        self.setReferences()

    def updateFrm(self):
        """
        Update the coordinate frame based on the current conformer.
        """
        pos = [x.GetPositions() for x in self.conformers]
        self.frm.loc[:] = np.concatenate(pos, axis=0)

    def placeInitFrags(self):
        log_debug(
            f'Placing {len(self.conformers)} initiators into the cell...')

        tenth, threshold, = len(self.conformers) / 10., 0
        for index, conf in enumerate(self.conformers, start=1):
            conf.placeInitFrag()
            if index >= threshold:
                new_line = "" if index == len(self.conformers) else ", [!n]"
                log_debug(
                    f"{int(index / len(self.conformers) * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)
        self.logInitFragsPlaced()

    def hasClashes(self, gids):
        """
        Whether the atoms has clashes with existing atoms in the cell.

        :param gids list: golabal atom ids to check clashes against.
        :return bool: True if clashes are found.
        """
        frag_rows = [self.frm.loc[x] for x in gids]
        for row in frag_rows:
            clashes = self.dcell.getClashes(row,
                                            included=self.dcell.extg_gids,
                                            radii=self.df_reader.radii,
                                            excluded=self.df_reader.excluded)
            if clashes:
                return True
        return False

    def add(self, gids):
        """
        Update trajectory frame, add atoms to the atom cell and existing record.

        :param gids list: gids of the atoms to be added
        """
        self.updateFrm()
        self.dcell.atomCellUpdate(gids)
        self.dcell.addGids(gids)

    def remove(self, gids):
        """
        Remove atoms from the atom cell and existing record.

        :param gids list: gids of the atoms to be removed
        """
        self.dcell.atomCellRemove(gids)
        self.dcell.removeGids(gids)

    def logInitFragsPlaced(self):
        """
        Log the initiator fragments status after the first placements.
        """

        log_debug(
            f'{len(self.conformers)} initiators have been placed into the cell.'
        )
        if len(self.mols) == 1:
            return
        log_debug(
            f'({self.init_tf.pairDists().min():.2f} as the minimum pair distance)'
        )


class Fragment:
    """
    Class to set a portion of the conformer by rotating the dihedral angle.
    """

    def __str__(self):
        """
        Print the dihedral angle four-atom ids and the swing atom ids.
        """
        return f"{self.dihe}: {self.aids}"

    def __init__(self, dihe, conf, delay=False):
        """
        :param dihe list of dihedral atom ids: the dihedral that changes the
            atom position in this fragment.
        :param mol 'GrownMol': the GrownMol that this fragment belongs to
        :param delay bool: whether to delay the initialization of the fragment.
        """
        self.dihe = dihe  # dihedral angle four-atom ids
        self.conf = conf  # Conformer object this fragment belongs to
        self.aids = []  # Atom ids of the swing atoms
        self.gids = set()  # Global atom ids of the swing atoms
        self.pfrag = None  # Previous fragment
        self.nfrags = []  # Next fragments
        self.vals = []  # Available dihedral values candidates
        self.val = None  # Chosen dihedral angle value
        self.fval = True  # All dihedral values are available (new frag)
        if delay:
            return
        self.setUp()

    def setUp(self):
        """
        Set up the fragment.
        """
        self.resetVals()
        self.aids = self.conf.getSwingAtoms(*self.dihe)

    def resetVals(self):
        """
        Reset the dihedral angle values and state.
        """
        self.val, self.fval = None, True
        self.vals = list(np.linspace(0, 360, 36, endpoint=False))

    def getOwningMol(self):
        """
        Get the owning GrownMol object.

        :return GrownMol: get the molecule this fragment belongs to.
        """
        return self.conf.GetOwningMol()

    def copyInit(self, conf):
        """
        Copy the current initial fragment and all the fragments retrieved by it.
        The connections between all new fragments are established as well.

        :param mol GrownConf: the conformer object this initial fragment belongs to
        :return Fragment: the copied initial fragment.
        """
        ifrag = self.copy(conf)
        all_nfrags = [ifrag]
        while all_nfrags:
            frag = all_nfrags.pop()
            nfrags = [x.copy(conf) for x in frag.nfrags]
            frag.nfrags = nfrags
            for nfrag in nfrags:
                nfrag.pfrag = frag
            all_nfrags += nfrags
        return ifrag

    def copy(self, conf):
        """
        Copy the current fragment to a new one.

        :param conf GrownConf: the conformer object this fragment belongs to.
        :return Fragment: the copied fragment.
        """
        frag = Fragment(self.dihe, conf, delay=True)
        frag.aids = self.aids
        frag.gids = [conf.id_map[x] for x in frag.aids]
        frag.pfrag = self.pfrag
        frag.nfrags = self.nfrags
        # Another conformer may have different value and candidates
        frag.vals = self.vals[:]
        frag.val = self.val
        frag.val = self.fval
        return frag

    def setFragments(self):
        """
        Set up fragments by iteratively searching for rotatable bond path and
        adding them as next fragments.
        """
        # Finish up the ifrag setup and set next fragments
        frags = self.setNextFrags()
        # Set next fragments
        while frags:
            frag = frags.pop(0)
            nfrags = frag.setNextFrags()
            frags += nfrags
        # Set previous fragments
        for frag in self.fragments():
            for nfrag in frag.nfrags:
                nfrag.pfrag = frag

    def setNextFrags(self):
        """
        Set fragments by searching for rotatable bond path and adding them as
        next fragments.

        :return list of 'Fragment': fragment self and newly added fragments.
            (the atom ids of the current fragments changed)
        """
        dihes = self.getNewDihes()
        if not dihes:
            # This removes self frag out of the to_be_fragmentized list
            return []
        if not self.dihe:
            # This is an initial fragment with unknown dihedral angle
            self.dihe = dihes.pop(0)
            self.setUp()
        frags = [self] + [Fragment(x, self.conf) for x in dihes]
        for frag, nfrag in zip(frags[:-1], frags[1:]):
            frag.aids = sorted(set(frag.aids).difference(nfrag.aids))
            frag.nfrags.append(nfrag)
        return frags

    def getNewDihes(self):
        """
        Get a list of dihedral angles that travel along the fragment rotatable
        bond to one fragment atom ids so that the path has the most rotatable
        bonds.

        NOTE: If the current dihedral angle of this fragment is not set, the
        dihedral angle list travel along the path with the most rotatable bonds.

        :return list of list: each sublist has four atom ids.
        """
        if self.dihe:
            pairs = zip([self.dihe[1]] * len(self.aids), self.aids)
        else:
            pairs = self.getOwningMol().findHeadTailPair()

        dihes, num = [], 0
        for source, target in pairs:
            _, _, path = self.getOwningMol().findPath(source=source,
                                                      target=target)
            a_dihes = zip(path[:-3], path[1:-2], path[2:-1], path[3:])
            a_dihes = [
                x for x in a_dihes if self.getOwningMol().isRotatable(x[1:-1])
            ]
            if len(a_dihes) < num:
                continue
            num = len(a_dihes)
            dihes = a_dihes
        return dihes

    def setDihedralDeg(self, val=None):
        """
        Set the dihedral angle of the current fragment. If val is not provided,
        randomly choose one from the candidates.

        :param val float: the dihedral angle value to be set.
        """
        if val is None:
            val = np.random.choice(self.vals)
            self.vals.remove(val)
            self.fval = False
        self.val = val
        self.conf.setDihedralDeg(self.dihe, self.val)

    def getPreAvailFrag(self):
        """
        Get the first previous fragment that has available dihedral candidates.

        :return 'Fragment': The previous fragment found.
        """
        frag = self.pfrag
        if frag is None:
            return None
        while frag.pfrag and not frag.vals:
            frag = frag.pfrag
        if frag.pfrag is None and not frag.vals:
            # FIXME: Failed conformer search should try to reduce clash criteria
            return None
        return frag

    def getNxtFrags(self):
        """
        Get the next fragments that don't have full dihedral value candidates.

        :return list: list of next fragments
        """
        frags, nfrags = [], [self]
        while nfrags:
            nfrags = [y for x in nfrags for y in x.nfrags if not y.fval]
            frags += nfrags
        return frags

    def reportStatus(self, frags):
        """
        Report the growing and failed molecule status.

        :param frags list of 'fragments.Fragment': the growing fragments
        """

        cur_mol_num = len(set([x.conf.GetId() for x in frags]))
        if cur_mol_num == self.mol_num:
            # No change of the growing molecule number from previous report
            return
        import pdb
        pdb.set_trace()
        self.mol_num = cur_mol_num
        finished_num = len(self.conformers) - self.mol_num
        log_debug(f'{finished_num} finished; {self.failed_num} failed.')
        return cur_mol_num

    def fragments(self):
        """
        Return all fragments.

        :return list of Fragment: all fragment of this conformer.
        """
        frags, nfrags = [], [self]
        while nfrags:
            frags += nfrags
            nfrags = [y for x in nfrags for y in x.nfrags]
        return frags
