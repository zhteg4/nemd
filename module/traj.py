import math
import oplsua
import itertools
import numpy as np
import pandas as pd


class Frame(pd.DataFrame):
    """
    Class to hold coordinate information.
    """

    BOX = 'box'
    SPAN = 'span'
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
        if name is None and xyz is not None:
            index = range(1, xyz.shape[0] + 1)
        else:
            index = None
        super().__init__(data=xyz, index=index, columns=self.UXYZ)
        self.setBox(box)

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

    def getXYZ(self, atom_id):
        """
        Get the XYZ of the atom id.

        :param atom_id int: atom id
        :return row (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        """
        return self.loc[atom_id]

    def setBox(self, box):
        """
        Set the box span from box limits.

        :param box str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        """
        self.attrs[self.BOX] = box
        self.attrs[self.SPAN] = {x: np.inf for x in self.UXYZ}
        if box is None:
            return
        for idx, col in enumerate(self.UXYZ):
            self.attrs[self.SPAN][col] = box[idx * 2 + 1] - box[idx * 2]

    def getDists(self, ids, xyz):
        """
        Get the distance between the xyz and the of the xyzs associated with the
        input atom ids.

        :param atom_id int: atom ids
        :param xyz (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        :return list of floats: distances
        """
        dists = self.getXYZ(ids) - xyz
        for col in self.UXYZ:
            dists[col] = dists[col].apply(
                lambda x: math.remainder(x, self.attrs[self.SPAN][col]))
        return np.linalg.norm(dists, axis=1)


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
        self.span = None
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
        Cells with separation distances less than the cutoff are set as neighbors.
        """

        def separation_dist(ijk):
            separation_ids = [y - 1 if y else y for y in ijk]
            return np.linalg.norm(self.grids * separation_ids)

        max_ids = [math.ceil(self.cut / x) + 1 for x in self.grids]
        ijks = itertools.product(*[range(max_ids[x]) for x in range(3)])
        # Adjacent Cells are zero distance separated.
        neigh_ids = [x for x in ijks if separation_dist(x) < self.cut]
        # Keep itself (0,0,0) cell as multiple atoms may be in one cell.
        signs = itertools.product((-1, 1), (-1, 1), (-1, 1))
        signs = [np.array(x) for x in signs]
        self.neigh_ids = set([tuple(y * x) for x in signs for y in neigh_ids])

    def setAtomCell(self):
        """
        Put atom ids into the corresponding cells.
        """
        ids = ((self.frm) / self.grids).round().astype(int) % self.indexes
        self.atom_cell = np.zeros((*self.indexes, ids.shape[0] + 1),
                                  dtype=bool)
        for row in ids.itertuples():
            self.atom_cell[row.xu, row.yu, row.zu][row.Index] = True

    def getNeighbors(self, xyz):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself)

        :param xyz 1x3 array of floats: xyz of one atom coordinates
        :return list int: the atom ids of the neighbor atoms
        """

        id = (xyz / self.grids).round().astype(int)
        ids = [tuple((id + x) % self.indexes) for x in self.neigh_ids]
        return [y for x in ids for y in self.atom_cell[x].nonzero()[0]]

    def getClashes(self,
                   row,
                   included=None,
                   excluded=None,
                   radii=None,
                   threshold=1.):
        """
        Get the clashes between xyz and atoms in the frame.

        :param row (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        :param included list of int: the atom ids included for the clash check
        :param excluded list of int: the atom ids excluded for the clash check
        :param radii dict: the values are the radii smaller than which are clashes
        :param threshold clash radii: clash criteria when radii not defined
        :return list of tuple: clashed atom ids, distance, and threshold
        """

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
        dists = self.frm.getDists(neighbors, xyz)
        if radii:
            thresholds = [radii[row.name][x] for x in neighbors]
        else:
            thresholds = [threshold] * len(neighbors)
        clashes = [(row.name, x, y, z)
                   for x, y, z in zip(neighbors, dists, thresholds) if y < z]
        return clashes
