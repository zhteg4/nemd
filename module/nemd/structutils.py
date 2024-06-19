# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles molecular topology and structural editing.
"""
import math
import scipy
import warnings
import itertools
import functools
import numpy as np
import networkx as nx
from rdkit import Chem
from scipy.spatial.transform import Rotation

from nemd import traj
from nemd import pnames
from nemd import logutils
from nemd import structure
from nemd import lammpsdata

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):
    """
    Print this message into the log file in debug mode.
    :param msg str: the msg to be printed
    """
    if logger is None:
        return
    logger.debug(msg)


class GriddedConf(lammpsdata.Conformer):
    ...


class ConfError(RuntimeError):
    """
    When max number of the failure for this conformer has been reached.
    """
    pass


class PackedConf(lammpsdata.Conformer):

    MAX_TRIAL_PER_CONF = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id_map = None
        self.frm = None
        self.dcell = None
        self.radii = None
        self.excluded = None

    def setReferences(self):
        """
        Set the references to the conformer.
        """
        self.radii = self.mol.radii
        self.excluded = self.mol.excluded
        self.frm = self.mol.frm
        self.dcell = self.mol.dcell

    def setConformer(self, max_trial=MAX_TRIAL_PER_CONF):
        """
        Place molecules one molecule into the cell without clash.

        :param max_trial int: the max trial number for each molecule to be
            placed into the cell.
        :raise ConfError: if the conformer always has clashes with the existing
            atoms in the cell after the maximum trial.
        """
        for _ in range(max_trial):
            self.translate(-np.array(self.centroid()))
            self.rotateRandomly()
            pnt = self.frm.getPoint()
            self.translate(pnt)
            self.updateFrm()
            if self.hasClashes():
                continue
            self.updateDcell()
            return
        raise ConfError

    def rotateRandomly(self, seed=None):
        """
        Randomly rotate the conformer.

        NOTE: the random state is set according to the numpy random seed.
        :param seed: the random seed to generate the rotation matrix.
        """
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        mtrx = np.identity(4)
        mtrx[:-1, :-1] = Rotation.random(random_state=seed).as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def rotate(self, ivect, tvect):
        """
        Rotate the conformer by three initial vectors and three target vectors.

        :param ivect 3x3 'numpy.ndarray': Each row is one initial vector
        :param tvect 3x3 'numpy.ndarray': Each row is one corresponding target
            vector
        """
        mtrx = np.identity(4)
        rotation, _ = Rotation.align_vectors(tvect, ivect)
        mtrx[:-1, :-1] = rotation.as_matrix()
        Chem.rdMolTransforms.TransformConformer(self, mtrx)

    def hasClashes(self, aids=None):
        """
        Whether the conformer has any clashes with the existing atoms.

        :param aids: the conformer atom ids.
        :return bool: True if clashes are found.
        """
        aids = self.aids if aids is None else aids
        gids = self.id_map[aids]
        values = self.frm.vloc(gids)
        for name, xyz in zip(gids, values):
            clashes = self.dcell.getClashes(xyz,
                                            name=name,
                                            included=self.dcell.extg_gids,
                                            radii=self.radii,
                                            excluded=self.excluded)
            if clashes:
                return True
        return False

    def updateFrm(self, aids=None):
        """
        Update the coordinate frame based on the current conformer.

        :param aids list: the atom ids whose coordinates are used.
        """
        if aids is None:
            aids = self.aids
        self.frm.update(self.id_map[aids], self.GetPositions()[aids, :])

    def updateDcell(self, aids=None):
        """
        Update the distance cell based on the current conformer.

        :param aids list: the atom ids to be updated with.
        """
        if aids is None:
            aids = self.aids
        self.dcell.atomCellUpdate(self.id_map[aids])
        self.dcell.addGids(self.id_map[aids])


class GrownConf(PackedConf):

    MAX_TRIAL_PER_CONF = 5

    def __init__(self, *args, **kwargs):
        super(GrownConf, self).__init__(*args, **kwargs)
        self.ifrag = None
        self.init_aids = None
        self.failed_num = 0
        self.frags = []
        self.oxyz = self.GetPositions()

    def reset(self):
        """
        Rest the attributes that are changed during one grow attempt.
        """
        self.failed_num = 0
        self.frags = [self.ifrag]
        if self.ifrag:
            self.ifrag.reset()
        self.setPositions(self.oxyz)

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

    @functools.cache
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
            self.updateFrm()
            if self.hasClashes(self.init_aids):
                continue
            self.updateDcell(self.init_aids)
            self.init_tf.loc[self.gid] = point
            return
        # with open('placeInitFrag.xyz', 'w') as out_fh:
        #     self.frm.write(out_fh,
        #                    dreader=self.lmw,
        #                    visible=list(self.dcell.extg_gids),
        #                    points=points)
        msg = f'Only {len(self.dcell.extg_gids)} / {len(self.dcell.gids)} placed'
        log_debug(msg)
        raise ConfError(msg)

    def setFrag(self, max_trial=MAX_TRIAL_PER_CONF):
        """
        Set part of the conformer by rotating the dihedral angle, back moving,
        and relocation.

        :param max_trial int: the max number of trials for one conformer.
        :raise ConfError: if the max number of trials for this conformer is
            reached.
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

        if self.failed_num > max_trial:
            msg = f'Placed {len(self.dcell.extg_gids)} / {len(self.dcell.gids)} ' \
                  f'atoms reaching max trial number for conformer {self.gid}.'
            log_debug(msg)
            raise ConfError
        self.failed_num += 1
        self.ifrag.reset()
        # The method backmove() deletes some extg_gids
        self.dcell.resetGraph()
        self.placeInitFrag()
        self.reportRelocation()

    def setDihedral(self, frag):
        """
        Set part of the conformer by rotating the dihedral angle.

        :param frag 'fragments.Fragment': fragment to set the dihedral angle.
        :return bool: True if successfully place one fragment.
        """
        while frag.vals:
            frag.setDihedralDeg()
            self.updateFrm()
            if self.hasClashes(frag.aids):
                continue
            self.updateDcell(frag.aids)
            self.frags += frag.nfrags
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
        ratom_aids = [y for x in nxt_frags for y in x.aids]
        if not found:
            ratom_aids += frag.conf.init_aids
        self.remove(ratom_aids)
        # 3）Fragment after the next fragments were added to the growing
        # frags before this backmove step.
        nnxt_frags = [y for x in nxt_frags for y in x.nfrags]
        self.frags = [frag] + [x for x in self.frags if x not in nnxt_frags]
        return found

    def remove(self, aids):
        """
        Remove atoms from the atom cell and existing record.

        :param aids list: aids of the atoms to be removed
        """
        self.dcell.atomCellRemove(self.id_map[aids])
        self.dcell.removeGids(self.id_map[aids])

    def reportRelocation(self):
        """
        Report the status after relocate an initiator fragment.
        """

        idists = self.init_tf.pairDists()
        dists = self.dcell.getDistsWithIds(self.id_map[self.init_aids])
        log_debug(f"Relocate the initiator of {self.gid} conformer "
                  f"(initiator: {idists.min():.2f}-{idists.max():.2f}; "
                  f"close contact: {dists.min():.2f}) ")
        log_debug(
            f'{len(self.dcell.extg_gids)} / {len(self.dcell.gids)} atoms placed.'
        )


class Mol(lammpsdata.Mol):

    def adjustBondLength(self):
        """
        Adjust bond length according to the force field parameters.
        """
        # Set the bond lengths of one conformer
        tpl = self.GetConformer()
        for bond in self.GetBonds():
            bonded = [bond.GetBeginAtom(), bond.GetEndAtom()]
            aids = set([x.GetIdx() for x in bonded])
            bond_type = self.rvrs_bonds[tuple(sorted(aids))]
            dist = self.ff.bonds[bond_type].dist
            tpl.setBondLength([x.GetIdx() for x in bonded], dist)
        # Update all conformers
        xyz = tpl.GetPositions()
        for conf in self.GetConformers():
            conf.setPositions(xyz)


class GriddedMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol to handle gridded conformers.
    """
    ConfClass = GriddedConf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # size = xyz span + buffer
        self.buffer = np.array([4, 4, 4])
        # The number of molecules per box edge
        self.mol_num = np.array([1, 1, 1])
        # The xyz shift within one box
        self.vecs = []

    def setConformers(self, vectors):
        """
        Place the conformers into boxes based on the shifting vectors.

        :param list of vectors np.ndarray: the translational vectors to move the
            conformer by multiple boxes distances.
        """
        mol_num = np.prod(self.mol_num)
        for box_id in range(self.box_num):
            vector = vectors.pop()
            for conf, vec in zip(self.confs[box_id * mol_num:], self.vecs):
                vecs = vector + vec
                conf.translate(vecs)

    def setConfNumPerEdge(self, size):
        """
        Set the number of molecules per edge of the box.

        :param size np.ndarray: the box size (the largest molecule size) to
            place this conformer in.
        """
        self.mol_num = np.floor(size / self.size).astype(int)

    @property
    def size(self):
        """
        Return the box size of this molecule.
        """
        # Grid layout assumes all conformers from one molecule are the same
        xyzs = self.GetConformer().GetPositions()
        return (xyzs.max(axis=0) - xyzs.min(axis=0)) + self.buffer

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

    @property
    def box_num(self):
        """
        Return the number of boxes (the largest molecule size) needed to place
            all conformers.
        """
        return math.ceil(self.GetNumConformers() / np.prod(self.mol_num))


class PackedMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    ConfClass = PackedConf

    def __init__(self, *args, **kwargs):
        """
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(*args, **kwargs)
        self.lmw = None
        self.frm = None
        self.dcell = None


class GrownMol(PackedMol):

    ConfClass = GrownConf
    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    PATT = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')
    IS_MONO = pnames.IS_MONO
    MONO_ID = pnames.MONO_ID
    POLYM_HT = pnames.POLYM_HT
    EDGES = 'edges'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ifrag = None
        self.init_aids = None
        self.graph = None
        self.rotatable_bonds = None
        if self.delay:
            return
        self.setGraph()
        self.rotatable_bonds = self.GetSubstructMatches(self.PATT,
                                                        maxMatches=1000000)

    def setGraph(self):
        """
        Get the networkx graph on the molecule.
        """
        self.graph = nx.Graph()
        edges = [[x.GetBeginAtom(), x.GetEndAtom()] for x in self.GetBonds()]
        edges = [tuple([x[0].GetIdx(), x[1].GetIdx()]) for x in edges]
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
                    node[self.EDGES].append(edge)
                except KeyError:
                    node[self.EDGES] = [edge]

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


class Struct(lammpsdata.Struct):

    def __init__(self, *args, **kwargs):
        """
        :param ff 'OplsParser': the force field class.
        :param options 'argparse.Namespace': command line options
        """
        super().__init__(*args, **kwargs)
        self.density = None
        self.lmw = None
        self.box = None

    def adjustBondLength(self):
        """
        Adjust the bond lenangths of all conformers.
        """
        for mol in self.molecules:
            mol.adjustBondLength()


class GriddedStruct(Struct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    MolClass = GriddedMol

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        Set the over-all periodic boundary box.
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
        vectors = [x * self.size for x in itertools.product(*idxes)]
        np.random.shuffle(vectors)
        # boxes are filled in random order with all molecules in random order
        molecules = self.molecules[:]
        np.random.shuffle(molecules)
        for mol in molecules:
            mol.setConformers(vectors)

    def setDensity(self):
        """
        Set the density of the structure.
        """
        vol = np.prod(self.box[1::2])
        vol *= math.pow(scipy.constants.centi / scipy.constants.angstrom, 3)
        self.density = self.mw * scipy.constants.Avogadro / vol


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class PackedStruct(Struct):
    """
    Pack molecules by random rotation and translation.
    """

    MolClass = PackedMol
    MAX_TRIAL_PER_DENSITY = 50

    def __init__(self, *args, **kwargs):
        """
        :param options 'argparse.Namespace': command line options
        """
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        super().__init__(*args, **kwargs)
        self.frm = None
        self.dcell = None

    def runWithDensity(self, density):
        """
        Create amorphous cell of the target density by randomly placing
        molecules with random orientations.

        NOTE: the final density of the output cell may be smaller than the
        target if the max number of trial attempt is reached.

        :param density float: the target density
        """
        # self.density is initialized in Struct.__init__() method
        self.density = density
        self.run()

    def run(self):
        """
        Create amorphous cell by randomly placing molecules with random
        orientations.
        """
        self.setBox()
        self.setTypeMap()
        self.setClashParams()
        self.adjustBondLength()
        self.setFrameAndDcell()
        self.setReferences()
        self.setConformers()

    def setBox(self):
        """
        Set periodic boundary box size.
        """
        vol = self.mw / self.density / scipy.constants.Avogadro
        edge = math.pow(vol, 1 / 3)  # centimeter
        edge *= scipy.constants.centi / scipy.constants.angstrom
        self.box = [0, edge, 0, edge, 0, edge]
        log_debug(f'Cubic box of size {edge:.2f} angstrom is created.')

    def setFrameAndDcell(self, **kwargs):
        """
        Set the trajectory frame and distance cell.
        """
        id = [x for x in range(1, self.atom_total + 1)]
        self.frm = traj.Frame(xyz=self.getPositions(), index=id, box=self.box)
        self.dcell = traj.DistanceCell(self.frm, **kwargs)
        self.dcell.setUp()

    def setReferences(self):
        """
        Set references to all molecular and conformers.
        """

        for mol in self.molecules:
            mol.radii = self.radii
            mol.excluded = self.excluded
            mol.frm = self.frm
            mol.dcell = self.dcell
            for conf in mol.GetConformers():
                conf.setReferences()

    def setConformers(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Place all molecules into the cell at certain density.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached or the chance of achieving the goal is too low.
        """
        trial_id, conf_num, finished, nth = 1, len(self.conformers), [], -1
        for trial_id in range(1, max_trial + 1):
            self.dcell.reset()
            for conf_id, conf in enumerate(self.conformers):
                try:
                    conf.setConformer()
                except ConfError:
                    break
                # One conformer successfully placed
                if nth != math.floor((conf_id + 1) / conf_num * 10):
                    # Progress
                    nth = math.floor((conf_id + 1) / conf_num * 10)
                    nline = "" if conf_id == conf_num - 1 else ", [!n]"
                    log_debug(f"{int((conf_id + 1) / conf_num * 100)}%{nline}")
                # Whether all molecules successfully placed
                if conf_id == conf_num - 1:
                    return
            # Current conformer failed
            log_debug(f'{trial_id} trail fails.')
            log_debug(f'Only {conf_id} / {conf_num} molecules placed.')
            finished.append(conf_id)
            if not bool(trial_id % int(max_trial / 10)):
                delta = conf_num - np.average(finished)
                std = np.std(finished)
                if not std:
                    raise DensityError
                zscore = abs(delta) / std
                if scipy.stats.norm.cdf(-zscore) * max_trial < 1:
                    # With successful conformer number following norm
                    # distribution, max_trial won't succeed for one time
                    raise DensityError
        raise DensityError


class GrownStruct(PackedStruct):

    MolClass = GrownMol
    MAX_TRIAL_PER_DENSITY = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_tf = None

    def run(self):
        """
        Create amorphous cell by randomly placing initiators of the conformers,
        and grow the conformers by adding fragments one by one.
        """
        self.setDataReader()
        self.setBox()
        self.updateConformers()
        self.setFrameAndDcell()
        self.setReferences()
        self.fragmentize()
        self.setConformers()

    def setFrameAndDcell(self):
        """
        Set distance cell parameters.
        """
        # memory saving flaot16 to regular float32
        # Using [0][1][2] as the cell, atoms in [0] and [2], are at least
        # Separated by 1 max_clash_dist, meaning no clashes.
        cut = float(self.lmw.radii.max())
        super().setFrameAndDcell(cut=cut)
        self.dcell.setGraph(len(self.conformers))
        data = np.full((len(self.conformers), 3), np.inf)
        index = [x.gid for x in self.conformers]
        self.init_tf = traj.Frame(xyz=data, index=index, box=self.box)

    def setReferences(self):
        """
        See parent class for details.
        """
        for mol in self.molecules:
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

    def placeInitFrags(self):
        """
        Place the initiators into cell.
        """
        log_debug(f'Placing {len(self.conformers)} initiators...')

        tenth, threshold, = len(self.conformers) / 10., 0
        for index, conf in enumerate(self.conformers, start=1):
            conf.placeInitFrag()
            if index >= threshold:
                new_line = "" if index == len(self.conformers) else ", [!n]"
                log_debug(
                    f"{int(index / len(self.conformers) * 100)}%{new_line}")
                threshold = round(threshold + tenth, 1)

        log_debug(f'{len(self.conformers)} initiators have been placed.')
        if len(self.conformers) == 1:
            return
        dist = self.init_tf.pairDists().min()
        log_debug(f'({dist:.2f} as the minimum pair distance)')

    def setConformers(self, max_trial=MAX_TRIAL_PER_DENSITY):
        """
        Looping conformer one by one and set one fragment configuration each
        time until all to full length.

        :param max_trial int: the max number of trials at one density.
        :raise DensityError: if the max number of trials at this density is
            reached.
        """
        log_debug("*" * 10 + f" {self.density} " + "*" * 10)

        for _ in range(max_trial):
            self.reset()
            conformers = self.conformers[:]
            while conformers:
                conf = conformers.pop(0)
                try:
                    conf.setFrag()
                except ConfError:
                    # Reset and try again as this conformer cannot be placed.
                    # 1 ) ifrag cannot be place; 2 ) failed_num reached maximum
                    break
                # Successfully set one fragment of this conformer.
                if conf.frags:
                    # The conformer has more fragments to grow.
                    conformers.append(conf)
                    continue
                # Successfully placed all fragments of one conformer
                finished_num = len(self.conformers) - len(conformers)
                failed_num = sum([x.failed_num for x in self.conformers])
                log_debug(f'{finished_num} finished; {failed_num} failed.')
                if not conformers:
                    # Successfully placed all conformers.
                    return
        # Max trial reached at this density.
        raise DensityError

    def reset(self):
        """
        Reset the state so that a new growing attempt can happen.
        """
        ...
        for conf in self.conformers:
            conf.reset()
        self.dcell.reset()
        self.placeInitFrags()


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

    def reset(self):
        """
        Reset the state of all fragments.
        """
        for frag in self.fragments():
            frag.resetVals()

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
            mol = self.getOwningMol()
            _, _, path = mol.findPath(source=source, target=target)
            a_dihes = zip(path[:-3], path[1:-2], path[2:-1], path[3:])
            a_dihes = [x for x in a_dihes if mol.isRotatable(x[1:-1])]
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
