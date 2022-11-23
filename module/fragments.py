import copy
import random
import numpy as np
from rdkit import Chem
import structutils

class Fragment:

    def __repr__(self):
        return f"{self.dihe}: {self.atom_ids}"

    def __init__(self, dihe, fmol):
        self.dihe = dihe
        self.fmol = fmol
        self.atom_ids = []
        self.nfrags = []
        self.vals = list(np.linspace(0, 360, 36, endpoint=False))

    def getNewDihes(self):
        source = self.dihe[1] if self.dihe else None
        targets = self.atom_ids if self.atom_ids else [None]
        src, trgt, path, path_len = None, None, None, -1
        for target in targets:
            asrc, atrgt, apath = self.fmol.findLongPath(source=source, target=target)
            if path_len >= len(apath):
                continue
            src, trgt, path, path_len = asrc, atrgt, apath, len(apath)
        dihes = list(zip(path[:-3], path[1:-2], path[2:-1], path[3:]))
        dihes = [x for x in dihes if self.fmol.isRotatable(x[1:-1])]
        return dihes

    def setFrags(self):
        dihes = self.getNewDihes()
        nfrags = [Fragment(x, self.fmol) for x in dihes]
        if self.dihe:
            nfrags = [self] + nfrags
        else:
            self.dihe = dihes[0]
            self.setUp()
            nfrags[0] = self
        [x.setUp() for x in nfrags[1:]]
        for frag, nfrag in zip(nfrags[:-1], nfrags[1:]):
            frag.addFrag(nfrag)
        return nfrags if dihes else []

    def addFrag(self, nfrag):
        self.atom_ids = sorted(set(self.atom_ids).difference(nfrag.atom_ids))
        self.nfrags.append(nfrag)

    def setUp(self):
        self.setSwingAtoms()
        self.setFragAtoms()

    def setSwingAtoms(self):
        self.swing_atom_ids = self.fmol.getSwingAtoms(*self.dihe)

    def setFragAtoms(self):
        self.atom_ids = self.swing_atom_ids[:]

    def popVal(self):
        val = random.choice(self.vals)
        self.vals.remove(val)
        return val

    def setDihedralDeg(self):
        val = self.popVal()
        Chem.rdMolTransforms.SetDihedralDeg(self.fmol.conf, *self.dihe, val)

    # def markEdges(self, wt=0.0001):
    #     marked = []
    #     atom_ids_set = set(self.atom_ids)
    #     for node in self.atom_ids:
    #         edges = self.graph.nodes[node][structutils.EDGES]
    #         edges = [x for x in edges if [y for y in x if y != node][0] in atom_ids_set]
    #         marked += edges
    #         for edge in edges:
    #             self.graph.edges[edge][structutils.WEIGHT] = wt
    #     return marked
    #
    # def clearEdges(self, edges):
    #     for edge in edges:
    #         self.graph.edges[edge].pop(structutils.WEIGHT)



class FragMol:

    # https://ctr.fandom.com/wiki/Break_rotatable_bonds_and_report_the_fragments
    PATT = Chem.MolFromSmarts(
        '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]')

    def __init__(self, mol):
        self.mol = mol
        self.conf = self.mol.GetConformer(0)
        self.graph = structutils.getGraph(mol)
        self.rotatable_bonds = self.mol.GetSubstructMatches(self.PATT)
        self.init_frag = None

    def getSwingAtoms(self, *dihe):
        oxyz = self.conf.GetPositions()
        oval = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *dihe)
        Chem.rdMolTransforms.SetDihedralDeg(self.conf, *dihe, oval + 5)
        xyz = self.conf.GetPositions()
        changed = np.isclose(oxyz, xyz)
        Chem.rdMolTransforms.SetDihedralDeg(self.conf, *dihe, oval)
        return [i for i, x in enumerate(changed) if not all(x)]

    def isRotatable(self, bond):
        in_ring = self.mol.GetBondBetweenAtoms(*bond).IsInRing()
        single = tuple(sorted(bond)) in self.rotatable_bonds
        return not in_ring and single

    def findLongPath(self, source=None, target=None):
        return structutils.findLongPath(self.graph, source=source, target=target)

    def setFragments(self):
        self.init_frag = Fragment([], self)
        to_be_fragmentized = self.init_frag.setFrags()
        while(to_be_fragmentized):
            frag = to_be_fragmentized.pop(0)
            nfrags = frag.setFrags()
            to_be_fragmentized += nfrags

    def fragments(self):
        all_frags = []
        nfrags = [self.init_frag]
        while(nfrags):
            all_frags += nfrags
            nfrags = [y for x in nfrags for y in x.nfrags]
        return all_frags

    def getNumFrags(self):
        pass

    def getInitAtomIds(self):
        frags = self.fragments()
        atom_ids = [y for x in frags for y in x.atom_ids]
        atom_ids_set = set(atom_ids)
        assert len(atom_ids) == len(atom_ids_set)
        return [x for x in range(self.mol.GetNumAtoms()) if x not in atom_ids_set]


        # _, _, path = structutils.findLongPath(self.graph)
        # dihes = list(zip(path[:-3], path[1:-2], path[2:-1], path[3:]))
        # self.frags = [Fragment(x, self) for x in dihes]
        # [x.setUp() for x in self.frags]
        # for frag, nfrag in zip(self.frags[:-1], self.frags[1:]):
        #     frag.addFrag(nfrag)
        #
        # for frag in self.frags:
        #     dihes = frag.getNewDihes()
        #     nfrags = [Fragment(x, self) for x in dihes]
        #     [x.setUp() for x in nfrags]
        #     for frag, nfrag in zip([frag] + nfrags[:-1], nfrags):
        #         frag.addFrag(nfrag)
        # import pdb;pdb.set_trace()