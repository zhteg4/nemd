# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles molecular topology and structural editing.
"""
import math
import rdkit
import warnings
import itertools
import numpy as np
import networkx as nx
from nemd import pnames
from nemd import conformerutils

EDGES = 'edges'
WEIGHT = 'weight'


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


class Mol(rdkit.Chem.rdchem.Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol with additional attributes and methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf_id = 0

    def GetConformer(self, conf_id=None):
        """
        Get the conformer of the molecule.

        :param conf_id int: the conformer id to get.
        """
        if conf_id is None:
            conf_id = self.conf_id
        return super(Mol, self).GetConformer(conf_id)


class Struct:
    """
    A class to handle multiple molecules and their conformers.
    """

    MOL_ID = pnames.MOL_ID

    def __init__(self, mols, MolClass=Mol, **kwargs):
        self.mols = {i: MolClass(x) for i, x in enumerate(mols, start=1)}
        for mol_id, conf in enumerate(self.conformers, start=1):
            conf.SetIntProp(pnames.MOL_ID, mol_id)

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


class GridMol(Mol):
    """
    A subclass of rdkit.Chem.rdchem.Mol to handle gridded conformers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = np.array([4, 4, 4])
        # The number of molecules per box edge
        self.mol_num = np.array([1, 1, 1])
        # The shift within one box
        self.vecs = []

    @property
    def size(self):
        """
        Return the box size of this molecule.
        """
        xyzs = self.GetConformer().GetPositions()
        return (xyzs.max(axis=0) - xyzs.min(axis=0)) + self.buffer

    def setMolNumPerEdge(self, box):
        """
        Set the number of molecules per edge of the box.

        :param box np.ndarray: the box size to place this molecule in.
        """
        self.mol_num = np.floor(box / self.size).astype(int)

    @property
    def box_num(self):
        """
        Return the number of boxes needed to place all molecules.
        """
        return math.ceil(self.GetNumConformers() / np.prod(self.mol_num))

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
        """
        mol_num_per_box = np.prod(self.mol_num)
        for idx in range(min([self.GetNumConformers(), mol_num_per_box])):
            vecs = vector + self.vecs[idx]
            conformerutils.translation(self.GetConformer(), vecs)
            self.conf_id += 1


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
