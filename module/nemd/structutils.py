# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles molecular topology and structural editing.
"""
import networkx as nx

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
