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
from nemd import logutils
from nemd import conformerutils

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


def getGraph(mol):
    """
    Get the networkx graph on the input molecule.
    :param mol `rdkit.Chem.rdchem.Mol`: the input molecule with/without bonds

    :return `networkx.classes.graph.Graph`: graph with nodes and edges.
    """
    graph = nx.Graph()
    edges = [(
        x.GetBeginAtom().GetIdx(),
        x.GetEndAtom().GetIdx(),
    ) for x in mol.GetBonds()]
    if not edges:
        # When bonds don't exist, just add the atom.
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx())
        return graph
    # When bonds exist, add edges and the associated atoms, assuming atoms in
    # one molecule are bonded.
    graph.add_edges_from(edges)
    for edge in edges:
        for idx in range(2):
            node = graph.nodes[edge[idx]]
            try:
                node[EDGES].append(edge)
            except KeyError:
                node[EDGES] = [edge]
    return graph


def findPath(graph=None, mol=None, source=None, target=None, **kwarg):
    """
    Find the path in a molecule.

    :param graph 'networkx.classes.graph.Graph': molecular networkx graph
    :param mol `rdkit.Chem.rdchem.Mol`: molecule to find path on
    :param source int: the input source node
    :param target int: the input target node
    :return int, int, list: source node, target node, and the path inbetween
    """

    if graph is None:
        graph = getGraph(mol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shortest_path = nx.shortest_path(graph,
                                         source=source,
                                         target=target,
                                         **kwarg)

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


class PackConf(Conformer):

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
        self.id_map = self.mol.id_map
        self.radii = self.mol.radii
        self.excluded = self.mol.excluded
        self.frm = self.mol.frm
        self.dcell = self.mol.dcell
        self.extg_gids = self.mol.extg_gids

    @property
    def gids(self):
        """
        Return the global atom ids of this conformer.

        :return list of int: the global atom ids of this conformer.
        """
        return self.id_map[self.GetId()]

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


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf_id = 0
        self.confs = None

    def SetConformerId(self, conf_id):
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
            self.confs = {i: x for i, x in enumerate(self.GetConformers())}
        return self.confs[conf_id]

    def GetConformers(self, ConfClass=Conformer):
        """
        :param ConfClass (sub)class 'rdkit.Chem.rdchem.Conformer': the conformer
            class to instantiate conformers.
        :return list of conformers: the conformers of the molecule.
        """
        if self.confs is not None:
            return [x for x in self.confs.values()]
        confs = super().GetConformers()
        self.confs = {i: ConfClass(x, mol=self) for i, x in enumerate(confs)}
        return [x for x in self.confs.values()]


class GridMol(Mol):
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
        Return the number of boxes needed to place all molecules.
        """
        return math.ceil(self.GetNumConformers() / np.prod(self.mol_num))

    def setMolNumPerEdge(self, box):
        """
        Set the number of molecules per edge of the box.

        :param box np.ndarray: the box size to place this molecule in.
        """
        self.mol_num = np.floor(box / self.size).astype(int)

    def setVecs(self, box):
        """
        Set the translational vectors for this molecule so that this molecule
        can be placed in the given box.

        :param box np.ndarray: the box size to place this molecule in.
        """
        ptc = [np.linspace(-0.5, 0.5, x, endpoint=False) for x in self.mol_num]
        ptc = [x - x.mean() for x in ptc]
        self.vecs = [
            x * box for x in itertools.product(*[[y for y in x] for x in ptc])
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


class PackMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, ff=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff = ff
        self.id_map = None
        self.radii = None
        self.excluded = None
        self.frm = None
        self.dcell = None
        self.extg_gids = None

    def GetConformers(self, ConfClass=PackConf):
        """
        See parant class for details.
        """
        return super(PackMol, self).GetConformers(ConfClass=ConfClass)

    @property
    def molecular_weight(self):
        """
        The molecular weight of the polymer.

        :return float: the total weight.
        """
        return self.ff.molecular_weight(self)

    mw = molecular_weight


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """

    def __init__(self, mols, MolClass=Mol, **kwargs):
        self.mols = {i: MolClass(x) for i, x in enumerate(mols, start=1)}
        for mol_id, conf in enumerate(self.conformers, start=1):
            conf.SetId(mol_id)

    @property
    def conformers(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all molecules.
        """
        return [x for y in self.molecules for x in y.GetConformers()]

    @property
    def molecules(self):
        """
        Return all conformers of all molecules.

        :return list of rdkit.Chem.rdchem.Conformer: the conformers of all molecules.
        """
        return [x for x in self.mols.values()]


class GridStruct(Struct):
    """
    Grid the space and fill sub-cells with molecules as rigid bodies.
    """

    def __init__(self, *args, MolClass=GridMol, **kwargs):
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.box = np.zeros([3])

    def run(self):
        """
        Set conformers for all molecules.
        """
        self.setBox()
        self.setVectors()
        self.setConformers()

    def setBox(self):
        """
        Set the box as the maximum size over all molecules.
        """
        self.box = np.array([x.size for x in self.molecules]).max(axis=0)

    def setVectors(self):
        """
        Set translational vectors based on the box for all molecules.
        """
        for mol in self.molecules:
            mol.setMolNumPerEdge(self.box)
            mol.setVecs(self.box)

    def setConformers(self):
        """
        Set coordinates.
        """
        # vectors shifts molecules by the largest box size
        box_total = sum(x.box_num for x in self.molecules)
        idxs = range(math.ceil(math.pow(box_total, 1. / 3)))
        vectors = [x * self.box for x in itertools.product(idxs, idxs, idxs)]
        # boxes are filled in random order with all molecules in random order
        molecules = self.molecules[:]
        while molecules:
            mol = np.random.choice(molecules)
            np.random.shuffle(vectors)
            vector = vectors.pop()
            mol.setConformers(vector)
            if mol.conf_id == mol.GetNumConformers():
                molecules.remove(mol)


class DensityError(RuntimeError):
    """
    When max number of the failure at this density has been reached.
    """
    pass


class PackedCell(Struct):
    """
    Pack molecules by random rotation and translation.
    """

    MAX_TRIAL_PER_DENSITY = 100

    def __init__(self,
                 *args,
                 MolClass=PackMol,
                 ff=None,
                 options=None,
                 **kwargs):
        """
        :param polymers 'Polymer': one polymer object for each type
        :param options 'argparse.Namespace': command line options
        """
        # Force field -> Molecular weight -> Box -> Frame -> Distance cell
        MolClass = functools.partial(MolClass, ff=ff)
        super().__init__(*args, MolClass=MolClass, **kwargs)
        self.ff = ff
        self.options = options
        self.extg_gids = set()

    def run(self):
        """
        Create amorphous cell by randomly placing molecules with random
        orientations.
        """
        self.setBoxes()
        self.setDataReader()
        self.setFrameAndDcell()
        self.setMolStructRefs()
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
        self.setReferences()
        self.placeMols()

    def setBoxes(self):
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

    def placeMols(self, max_trial=MAX_TRIAL_PER_DENSITY):
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
