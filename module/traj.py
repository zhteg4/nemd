import numpy as np
import math
import oplsua
import itertools
import pandas as pd


class Frame(pd.DataFrame):
    """
    Class to hold coordinate information.
    """

    BOX = 'box'
    XU = 'xu'
    YU = 'yu'
    ZU = 'zu'
    UXYZ = [XU, YU, ZU]

    def __init__(self, xyz=None, box=None):
        """
        :param xyz nx3 'numpy.ndarray' or 'DataFrame':
        :param box str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        """

        try:
            name = xyz.values.index.name
        except AttributeError:
            name = None
        if name is not None and xyz is not None:
            index = range(1, xyz.shape[0] + 1)
        else:
            index = None
        super().__init__(data=xyz, index=index, columns=self.UXYZ)
        self.attrs[self.BOX] = box

    @classmethod
    def read(cls, filename):
        """
        Read a custom dumpy file with id, xu, yu, zu.

        :param filename str: the filename to read frames
        :return iterator of 'Frame': each frame has coordinates and box info
        """
        with open(filename, 'r') as dmp_fh:
            while True:
                lines = [dmp_fh.readline() for _ in range(9)]
                if not all(lines):
                    return
                atom_num = int(lines[3].strip('\n'))
                # 'id', 'xu', 'yu', 'zu'
                names = lines[-1].strip('\n').split()[-4:]
                frm = pd.read_csv(dmp_fh,
                                  nrows=atom_num,
                                  header=None,
                                  delimiter='\s',
                                  index_col=0,
                                  names=names,
                                  engine='python')
                if frm.shape[0] != atom_num or frm.isnull().values.any():
                    return
                # array([  8.8 ,  68.75,   2.86,  67.43, -28.76,  19.24])
                box = np.array([
                    float(y) for x in range(5, 8)
                    for y in lines[x].strip('\n').split()
                ])
                yield cls(frm, box=box)


class DistanceCell:
    """
    Class to quick search neighbors based on distance criteria and perform clash
    checking.
    """

    SCALE = oplsua.DataFileReader.SCALE
    BOX = Frame.BOX

    def __init__(self, frm=None, cut=6., resolution=2.):
        """
        :param frm 'Frame': coordinate frame
        :param cut float:
        :param resolution float:
        """
        self.frm = frm
        self.cut = cut
        self.resolution = resolution
        self.neigh_ids = None
        self.atom_cell = None

    def setUp(self):
        self.setSpan()
        self.setgrids()
        self.setNeighborIds()
        self.setAtomCell()

    def setSpan(self):
        """
        Set span based on PBCs.
        Span: the max PBC edge - the min PBC edge in each dimesion.
        """
        box = self.frm.attrs[self.BOX]
        self.span = np.array([box[i * 2 + 1] - box[i * 2] for i in range(3)])
        self.hspan = self.span / 2

    def setgrids(self):
        """
        Set grids and indexes.
        Indexes: the number of cells in three dimensions
        Grids: the length of the cell in each dimension
        """
        self.indexes = [math.ceil(x / self.resolution) for x in self.span]
        self.grids = np.array([x / i for x, i in zip(self.span, self.indexes)])

    def setNeighborIds(self):
        """
        Set neighbor cell ids. All cells with the distance smaller than
        the cut threshold are considered as each other's neighbor.
        """
        max_ids = [math.ceil(self.cut / x) for x in self.grids]
        ijks = itertools.product(*[range(max_ids[x]) for x in range(3)])
        ijks = {x: [y - 1 if y else y for y in x] for x in ijks}
        neigh_ids = [
            x for x,y in ijks.items()
            if np.linalg.norm(self.grids * y) <= self.cut
        ]
        # Don't remove the (0, 0, 0,) as multiple atom may be in one cell
        self.neigh_ids = set([
            tuple(np.array(ijk) * signs)
            for signs in itertools.product((-1, 1), (-1, 1), (-1, 1))
            for ijk in neigh_ids
        ])

    def setAtomCell(self, ):
        ids = ((self.frm) / self.grids).round().astype(int) % self.indexes
        self.atom_cell = np.zeros((*self.indexes, ids.shape[0] + 1),
                                  dtype=bool)
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
        neighbors = set(neighbors)
        if included:
            neighbors = neighbors.intersection(included)
        if excluded:
            neighbors = neighbors.difference(excluded[row.name])
        neighbors = list(neighbors)
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
