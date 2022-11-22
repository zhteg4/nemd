import networkx as nx


def getGraph(mol):
    graph = nx.Graph()
    edges = [(
        x.GetBeginAtom().GetIdx(),
        x.GetEndAtom().GetIdx(),
    ) for x in mol.GetBonds()]
    graph.add_edges_from(edges)
    return graph
