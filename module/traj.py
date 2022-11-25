import numpy as np
import math
import itertools
from bitsets import bitset
from bitarray import bitarray


class DistanceCell:

    def __init__(self, frm=None, box=None, cut=6., resolution=2.):
        self.frm = frm
        self.box = box
        self.cut = cut
        self.resolution = resolution
        self.neigh_ids = None
        self.atom_cell = None

    def setUp(self):
        self.setBox()
        self.setSpan()
        self.setgrids()
        self.setNeighborIds()
        self.setAtomCell()

    def setBox(self):
        if self.box is not None:
            return

        self.box = self.frm.attrs['box']

    def setSpan(self):
        self.span = np.array(
            [self.box[i * 2 + 1] - self.box[i * 2] for i in range(3)])
        self.hspan = self.span / 2

    def setgrids(self):
        self.indexes = [math.ceil(x / self.resolution) for x in self.span]
        self.grids = np.array([x / i for x, i in zip(self.span, self.indexes)])

    def setNeighborIds(self):
        max_ids = [math.ceil(self.cut / x) for x in self.grids]
        neigh_ids = [
            ijk for ijk in itertools.product(
                *[range(max_ids[x]) for x in range(3)])
            if math.dist((0, 0, 0), self.grids * ijk) <= self.cut
        ]
        # neigh_ids.remove((0, 0, 0,))
        self.neigh_ids = set([
            tuple(np.array(ijk) * signs)
            for signs in itertools.product((-1, 1), (-1, 1), (-1, 1))
            for ijk in neigh_ids
        ])

    def setAtomCell(self, ):
        ids = ((self.frm) / self.grids).round().astype(int) % self.indexes
        self.atom_cell = np.zeros((*self.indexes, ids.shape[0] + 1), dtype=np.bool)
        for row in ids.itertuples():
            self.atom_cell[row.xu, row.yu, row.zu][row.Index] = True

    def getNeighbors(self, xyz):

        id = (xyz / self.grids).round().astype(int)
        ids = [tuple((id + x) % self.indexes) for x in self.neigh_ids]
        return [y for x in ids for y in self.atom_cell[x].nonzero()[0]]

    def getClashes(self,
                   row,
                   included=None,
                   excluded=None,
                   radii=None,
                   threshold=1.):
        xyz = row.values
        neighbors = self.getNeighbors(xyz)
        try:
            neighbors.remove(row.name)
        except ValueError:
            pass
        if included:
            neighbors = [x for x in neighbors if x in included]
        if excluded:
            neighbors = [x for x in neighbors if x not in excluded[row.name]]
        dists = np.linalg.norm(
            (self.frm.loc[neighbors] - xyz + self.hspan) % self.span -
            self.hspan,
            axis=1)
        if radii:
            thresholds = [radii[row.name][x] for x in neighbors]
        else:
            thresholds = [threshold] * len(neighbors)
        clashes = [(row.name, x, y, z)
                   for x, y, z in zip(neighbors, dists, thresholds) if y < z]
        return clashes
