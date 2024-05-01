# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module read, parser, and analyze trajectories.

Unzip a GZ trajectory file: zip -dv dump.custom.gz
"""
import io
import os
import math
import gzip
import numba
import types
import random
import base64
import warnings
import itertools
import subprocess
import collections
import numpy as np
import pandas as pd
import networkx as nx
from contextlib import contextmanager

from nemd import oplsua
from nemd import symbols

FlAG_CUSTOM_DUMP = 'custom_dump'
FlAG_DATA_FILE = '-data_file'
ITEM_TIMESTEP = 'ITEM: TIMESTEP'


def frame_steps(filename):
    """
    Get the frame steps.

    :param filename str: the filename to read frames
    :return 'numpy.ndarray': the step information of all steps
    """
    info = subprocess.run(
        f"zgrep '{ITEM_TIMESTEP}' {filename} "
        f"--no-group-separator -A1 | grep -v '{ITEM_TIMESTEP}'",
        capture_output=True,
        shell=True)
    return np.loadtxt(io.StringIO(info.stdout.decode("utf-8")), dtype=int)


def slice_frames(filename=None, contents=None, slice=slice, start=0):
    """
    Get and slice the trajectory frames.

    :param filename str: the filename to read frames
    :param contents `bytes`: parse the contents if filename not provided
    :param slice list: start, stop, and interval
    :param start int: only frames with step >= this value will be fully read
    :return iterator of 'Frame': each frame has coordinates and box info
    """
    frm_iter = get_frames(filename=filename, contents=contents, start=start)
    if not slice:
        return frm_iter
    return itertools.islice(frm_iter, *slice)


def get_frames(filename=None, contents=None, start=0):
    """
    Get the trajectory frames based on file extension.

    :param filename str: the filename to read frames
    :param contents `bytes`: parse the contents if filename not provided
    :param start int: only frames with step >= this value will be fully read
    :return iterator of 'Frame': each frame has coordinates and box info
    """
    is_xyz = False
    if filename:
        is_xyz = filename.endswith('.xyz')
    if contents:
        content_type, content_string = contents.split(',')
        is_xyz = content_type.split(';')[0].endswith('x-xyz')
        contents = base64.b64decode(content_string).decode("utf-8")
    if is_xyz:
        return Frame.readXYZ(filename=filename, contents=contents)
    return Frame.read(filename=filename, contents=contents, start=start)


class Frame(pd.DataFrame):
    """
    Class to hold coordinate information.
    """

    BOX = 'box'
    STEP = 'step'
    SPAN = 'span'
    XU = 'xu'
    YU = 'yu'
    ZU = 'zu'
    XYZU = [XU, YU, ZU]
    ELEMENT = 'element'
    SIZE = 'size'
    COLOR = 'color'
    XYZU_ELE_SZ_CLR = XYZU + [ELEMENT, SIZE, COLOR]

    def __init__(self,
                 xyz=None,
                 box=None,
                 index=None,
                 columns=None,
                 step=None):
        """
        :param xyz nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param box str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        :param index list: the atom indexes
        :param columns list: the data columns (e.g., xu, yu, zu, element)
        :param step int: the number of simulation step that this frame is at
        """
        try:
            name = xyz.values.index.name
        except AttributeError:
            name = None
        if name is None and index is None and xyz is not None:
            index = range(1, xyz.shape[0] + 1)
        if columns is None:
            columns = self.XYZU
        super().__init__(data=xyz, index=index, columns=columns, dtype=float)
        self.setBox(box)
        self.setStep(step)

    @classmethod
    def read(cls, filename=None, contents=None, start=0):
        """
        Read a custom dumpy file with id, xu, yu, zu.

        Note: only fully read frames contain full information.

        :param filename str: the filename to read frames
        :param contents `bytes`: parse the contents if filename not provided
        :param start int: only frames with step >= this value will be fully read
        :return iterator of 'Frame' and/or 'SimpleNamespace': 'Frame' has step,
            coordinates and box information. 'SimpleNamespace' only has step info.
        """
        with cls.open_traj(filename=filename, contents=contents) as fh:
            while True:
                lines = [fh.readline() for _ in range(9)]
                if not all(lines):
                    return
                atom_num = int(lines[3].rstrip())
                step = int(lines[1].rstrip())
                if start > step:
                    with warnings.catch_warnings(record=True):
                        np.loadtxt(fh, skiprows=atom_num, max_rows=0)
                        frame = types.SimpleNamespace(step=step)
                else:
                    try:
                        data = np.loadtxt(fh, max_rows=atom_num)
                    except EOFError:
                        return
                    if data.shape[0] != atom_num:
                        return
                    data = data[data[:, 0].argsort()]
                    # array([  8.8 ,  68.75,   2.86,  67.43, -28.76,  19.24])
                    box = np.array([
                        float(y) for x in range(5, 8)
                        for y in lines[x].rstrip().split()
                    ])
                    # 'xu', 'yu', 'zu'
                    columns = lines[-1].rstrip().split()[-3:]
                    frame = cls(xyz=data[:, 1:],
                                box=box,
                                index=data[:, 0].astype(int),
                                columns=columns,
                                step=step)
                yield frame

    @classmethod
    def readXYZ(cls, filename=None, contents=None, box=None):
        """
        Read a xyz dumpy file with element, xu, yu, zu.

        :param filename str: the filename to read frames
        :param contents `bytes`: parse the contents if filename not provided
        :param box list: box of the frame (overwritten by the file header)
        :return iterator of 'Frame': each frame has coordinates and box info
        """
        with cls.open_traj(filename=filename, contents=contents) as fh:
            while True:
                line = fh.readline()
                if not line:
                    return
                atom_num = int(line.strip('\n'))
                names = [cls.ELEMENT] + cls.XYZU
                line = fh.readline().strip().split()
                if len(line) == 9:
                    box = [
                        float(z) for x, y in zip(line[1::3], line[2::3])
                        for z in [x, y]
                    ]
                frm = pd.read_csv(fh,
                                  nrows=atom_num,
                                  delimiter=r'\s',
                                  names=names,
                                  engine='python')
                frm.index = pd.RangeIndex(1, atom_num + 1)
                yield cls(frm, columns=cls.XYZU + [cls.ELEMENT], box=box)

    @staticmethod
    @contextmanager
    def open_traj(filename=None, contents=None):
        """
        Open trajectory file.

        :param filename: the filename with path
        :type filename: str
        :param contents: the trajectory contents
        :type contents: str
        :return: the file handle
        :rtype: '_io.TextIOWrapper'
        """
        if all([filename is None, contents is None]):
            raise ValueError(f'Please specify either filename or contents.')
        if filename:
            if os.path.isfile(filename):
                func = gzip.open if filename.endswith('.gz') else open
                fh = func(filename, 'rt')
            else:
                raise FileNotFoundError(f'{filename} not found')
        else:
            fh = io.StringIO(contents)
        try:
            yield fh
        finally:
            fh.close()

    def getPoint(self):
        """
        Get the XYZ of in the span.

        :param atom_id int: atom id
        :return row (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        """

        span = np.array([x for x in self.attrs[self.SPAN].values()])
        point = np.random.rand(3) * span
        point = [x + y for x, y in zip(point, self.attrs[self.BOX][::2])]

        return np.array(point)

    def getXYZ(self, atom_id):
        """
        Get the XYZ of the atom id.

        :param atom_id int: atom id
        :return row (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        """
        return self.loc[atom_id]

    def setStep(self, step):
        """
        Set the simulation step.

        :param step int: the number of simulation step that this frame is at
        """
        if step is None:
            return
        self.attrs[self.STEP] = step

    @property
    def step(self):
        """
        Get the simulation step.

        :param int: the number of simulation step that this frame is at
        """
        return self.attrs[self.STEP]

    def setBox(self, box):
        """
        Set the box span from box limits.

        :param box str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        """
        self.attrs[self.BOX] = box
        self.attrs[self.SPAN] = {x: np.inf for x in self.XYZU}
        if box is None:
            return
        for idx, col in enumerate(self.XYZU):
            self.attrs[self.SPAN][col] = box[idx * 2 + 1] - box[idx * 2]

    def getBox(self):
        """
        Set the box from box limits.

        :param str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        """
        return self.attrs[self.BOX]

    def getSpan(self):
        """
        Set the span from box limits.

        :param dict: {'xu': xxx, 'yu': xxx, 'zu': xxx} as the span
        """
        return self.attrs[self.SPAN]

    def getVolume(self):
        """
        Get the volume of the frame.

        :param float: the volume of the frame
        """
        return np.prod([x for x in self.attrs[self.SPAN].values()])

    def getDensity(self):
        """
        Get the number density of the frame.

        :param float: the number density of the frame
        """
        return self.shape[0] / self.getVolume()

    def getEdges(self):
        """
        Get the edges of the box.

        :return list of list: each sublist contains two points describing one
            edge.
        """
        box = self.getBox()
        if box is None:
            return []
        return oplsua.DataFileReader.getEdgesFromList(box)

    def getDists(self, ids, xyz, id_map=None, span=None):
        """
        Get the distance between the xyz and the of the xyzs associated with the
        input atom ids.

        :param atom_id int: atom ids
        :param xyz (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        :param id_map 'numpy.ndarray': the map from atom id to xyz row id
        :param span 'numpy.ndarray': the span of box
        :return list of floats: distances
        """
        if span is None:
            span = np.array(list(self.attrs[self.SPAN].values()))
        if id_map is None:
            dists = (self.getXYZ(ids) - xyz).values
        else:
            dists = self.values[id_map[ids], :] - np.array(xyz)
        return self.remainderIEEE(dists, span)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def remainderIEEE(dists, span):
        """
        Calculate IEEE 754 remainder.

        https://stackoverflow.com/questions/26671975/why-do-we-need-ieee-754-remainder

        :param dists numpy.ndarray: distances
        :param span numpy.ndarray: box span
        :return numpy.ndarray: distances within half box span
        """
        dists -= np.round(np.divide(dists, span)) * span
        return [np.linalg.norm(x) for x in dists]

    def pairDists(self, ids=None, cut=None, res=2.):
        """
        Get the distance between atom pair.

        :param ids list: list of gids as the atom selection
        :param cut float: the cutoff distance to search neighbors
        :param res float: the res of the grid step
        :return `numpy.ndarray`: distances array

        NOTE: sel.values is used instead of iterrows due to performace
        https://stackoverflow.com/questions/24870953/does-pandas-iterrows-have-performance-issues
        """
        ids = sorted(ids) if ids else list(range(1, self.shape[0] + 1))
        if cut:
            dcell = DistanceCell(self, gids=ids, cut=cut, res=res)
            dcell.setUp()
        id = {label: i for i, label in enumerate(self.index)}
        id_map = np.array([id.get(x, -1) for x in range(self.index.max() + 1)])
        span = np.array(list(self.attrs[self.SPAN].values()))
        sel, dists = self.loc[ids], []
        for idx, (id, row) in enumerate(zip(sel.index, sel.values)):
            oids = [x for x in dcell.getNeighbors(row)
                    if x > id] if cut else ids[idx + 1:]
            dist = self.getDists(oids, row, id_map=id_map, span=span)
            dists.append(dist)
        return np.concatenate(dists)

    def wrapCoords(self, broken_bonds=False, dreader=None):
        """
        Wrap coordinates into the PBC box. If broken_bonds is False and mols is
        provided, the geometric center of each molecule is wrapped into the box.

        :param broken_bonds bool: If True, bonds may be broken by PBC boundaries
        :param dreader 'oplsua.DataFileReader': to get molecule ids and
            associated atoms
        """
        if dreader is None:
            return

        span = np.array([x for x in self.attrs[self.SPAN].values()])
        if broken_bonds:
            self.loc[:] = self.loc[:] % span
            # The wrapped xyz shouldn't support molecule center operation
            return

        if dreader.mols is None:
            return

        # The unwrapped xyz can directly perform molecule center operation
        for mol in dreader.mols.values():
            center = self.loc[mol].mean(axis=0)
            delta = (center % span) - center
            self.loc[mol] += delta

    def glue(self, dreader=None):
        """
        Circular mean to compact the molecules. Good for molecules droplets in
        vacuum. (extension needed for droplets or clustering in solution)

        :param dreader 'oplsua.DataFileReader': to get molecule ids and
            associated atoms are available
        """
        if dreader is None:
            return

        span = np.array([x for x in self.attrs[self.SPAN].values()])
        centers = pd.concat(
            [self.loc[x].mean(axis=0) for x in dreader.mols.values()],
            axis=1).transpose()
        centers.index = dreader.mols.keys()
        theta = centers / span * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta = np.arctan2(sin_theta.mean(), cos_theta.mean())
        mcenters = theta * span / 2 / np.pi
        cshifts = ((mcenters - centers) / span).round() * span
        for id, mol in dreader.mols.items():
            cshift = cshifts.loc[id]
            self.loc[mol] += cshift

    def write(self, fh, dreader=None, visible=None, points=None):
        """
        Write XYZ to a file.

        :param fh '_io.TextIOWrapper': file handdle to write out xyz.
        :param dreader 'nemd.oplsua.DataFileReader': datafile reader for element info.
        :param visible list: visible atom gids.
        :param points list: additional point to visualize.
        """

        data = self.loc[visible] if visible else self
        if dreader is None:
            data.index = [symbols.UNKNOWN] * data.shape[0]
        else:
            data.index = [dreader.atoms[x].ele for x in data.index]
        box = self.getBox()
        header = [
            f'{j} {box[i*2]} {box[i*2+1]}'
            for i, j in enumerate(self.columns.to_list())
        ]
        if points:
            points = np.array(points)
            points = pd.DataFrame(points,
                                  index=['X'] * points.shape[0],
                                  columns=self.XYZU)
            data = pd.concat((data, points), axis=0)
        fh.write(f'{data.shape[0]}\n')
        data.to_csv(fh,
                    mode='a',
                    index=True,
                    sep=' ',
                    header=header,
                    quotechar=' ',
                    float_format='%.3f')


class DistanceCell:
    """
    Class to quick search neighbors based on distance criteria and perform clash
    checking.
    """

    SCALE = oplsua.DataFileReader.SCALE
    BOX = Frame.BOX
    AUTO = 'auto'
    INIT_NBR_INCR = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0),
                     (0, 0, -1)]

    def __init__(self, frm=None, gids=None, cut=6., res=AUTO):
        """
        :param frm 'Frame': trajectory frame
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distace to search neighbors
        :param res float: the res of the grid step
        """
        self.frm = frm
        self.cut = cut
        self.gids = gids
        self.res = res
        self.span = None
        self.neigh_ids = None
        self.atom_cell = None
        self.extg_gids = set()
        if self.gids is None:
            self.gids = list(range(1, self.frm.shape[0] + 1))

    def setUp(self):
        self.setSpan()
        self.setgrids()
        self.setNeighborIds()
        self.setNeighborMap()
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
        res = self.cut if self.res == self.AUTO else self.res
        self.indexes = [math.ceil(x / res) for x in self.span]
        self.indexes_numba = numba.int32(self.indexes)
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

    def setNeighborMap(self):
        """
        Set map between node id to neighbor node ids.
        """
        neigh_ids = np.array(list(self.neigh_ids))
        self.neigh_map = self.getNeighborMap(self.indexes_numba, neigh_ids)

    @staticmethod
    @numba.jit(nopython=True, parallel=True, cache=True)
    def getNeighborMap(indexes, neigh_ids):
        """
        Get map between node id to neighbor node ids.

        :param indexes numpy.ndarray: the number of cells in three dimensions
        :param neigh_ids numpy.ndarray: Neighbors cells (separation distances
            less than the cutoff)
        :return numpy.ndarray: map between node id to neighbor node ids
        """
        shape = (indexes[0], indexes[1], indexes[2], len(neigh_ids), 3)
        neigh_map = np.empty(shape, dtype=numba.int32)
        for xid in numba.prange(indexes[0]):
            for yid in numba.prange(indexes[1]):
                for zid in numba.prange(indexes[2]):
                    id = np.array([xid, yid, zid])
                    neigh_map[xid, yid, zid, :, :] = (neigh_ids + id) % indexes
        return neigh_map

    def setAtomCell(self):
        """
        Put atom ids into the corresponding cells.

        self.atom_cell.shape = [X index, Y index, Z index, all atom ids]
        """
        # ids = ((self.frm) / self.grids).round().astype(int) % self.indexes
        # self.atom_cell = np.zeros((*self.indexes, ids.shape[0] + 1),
        #                           dtype=bool)
        # for row in ids.loc[self.gids].itertuples():
        #     self.atom_cell[row.xu, row.yu, row.zu][row.Index] = True
        # The above code is sped up by the following
        atom_ids = numba.int32(self.frm.index)
        self.atom_cell = self.setAtomCellNumba(atom_ids, self.frm.values,
                                               self.grids, self.indexes_numba)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def setAtomCellNumba(atom_ids, xyzs, grids, indexes):
        """
        Put atom ids into the corresponding cells.

        :param atom_ids 'numpy.ndarray': the atom gids
        :param xyzs 'numpy.ndarray': xyz of atom coordinates
        :param grids 'numpy.ndarray': the length of the cell in each dimension
        :param indexes list of numba.int32: the number of the cell in each dimension
        :return 'numpy.ndarray': map between cell id to atom ids
            [X index, Y index, Z index, all atom ids]
        """

        cids = np.round(xyzs / grids).astype(numba.int32) % indexes
        shape = (indexes[0], indexes[1], indexes[2], cids.shape[0] + 1)
        atom_cell = np.zeros(shape, dtype=numba.boolean)
        for aid, cid in zip(atom_ids, cids):
            atom_cell[cid[0], cid[1], cid[2]][aid] = True
        return atom_cell

    def atomCellUpdate(self, gids):
        ids = ((self.frm.loc[gids]) /
               self.grids).round().astype(int) % self.indexes
        for row in ids.itertuples():
            self.atom_cell[row.xu, row.yu, row.zu][row.Index] = True

    def atomCellRemove(self, gids):
        ids = ((self.frm.loc[gids]) /
               self.grids).round().astype(int) % self.indexes
        for row in ids.itertuples():
            self.atom_cell[row.xu, row.yu, row.zu][row.Index] = False

    def getNeighbors(self, xyz):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself)

        :param xyz 1x3 array of floats: xyz of one atom coordinates
        :return list int: the atom ids of the neighbor atoms
        """
        return self.getNeighborsNumba(xyz, self.grids, self.indexes_numba,
                                      self.neigh_map, self.atom_cell)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def getNeighborsNumba(xyz, grids, indexes, neigh_map, atom_cell):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself) via Numba.

        :param xyz 1x3 'numpy.ndarray': xyz of one atom coordinates
        :param grids 'numpy.ndarray': the length of the cell in each dimension
        :param indexes list of 'numba.int32': the number of the cell in each dimension
        :param neigh_map ixjxkxnx3 'numpy.ndarray': map between cell id to neighbor cell ids
        :param atom_cell ixjxkxn array of floats: map cell id into containing atom ids
        :return list int: the atom ids of the neighbor atoms
        """
        # The cell id for xyz
        id = np.round(xyz / grids).astype(numba.int32) % indexes
        # Unique neighbor cell ids
        ids = neigh_map[id[0], id[1], id[2], :]
        mx = [np.max(ids[:, i]) + 1 for i in range(3)]
        uids = np.zeros((mx[0], mx[1], mx[2]), dtype=numba.boolean)
        for x in ids:
            uids[x[0], x[1], x[2]] = True
        # The atom ids from all neighbor cells
        neighbors = [
            j for i, x in np.ndenumerate(uids) if x
            for j in atom_cell[i[0], i[1], i[2], :].nonzero()[0]
        ]
        return neighbors

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
        # For small box, the same neighbor across PBCs appears multiple times
        neighbors = set(neighbors)
        try:
            neighbors.remove(row.name)
        except KeyError:
            pass
        if included is not None:
            neighbors = neighbors.intersection(included)
        if excluded is not None:
            neighbors = neighbors.difference(excluded[row.name])
        if not neighbors:
            return
        neighbors = list(neighbors)
        dists = self.frm.getDists(neighbors, xyz).round(4)
        if radii is None:
            thresholds = [threshold] * len(neighbors)
        else:
            thresholds = [radii[row.name][x] for x in neighbors]
        clashes = [(row.name, x, y, z)
                   for x, y, z in zip(neighbors, dists, thresholds) if y < z]
        return clashes

    def removeGids(self, gids):
        self.extg_gids = self.extg_gids.difference(gids)

    def addGids(self, gids):
        self.extg_gids.update(gids)

    def setGraph(self, mol_num, min_num=1000):
        """
        Set graph using grid intersection as nodes and connect neighbor nodes.

        :param mol_num int: molecule number.
        :param min_num int: minimum number of sites.
        """
        self.graph = nx.Graph()
        # getVoids() doesn't generate enough voids with scaling down the grid
        mgrid = pow(np.prod(self.span) / max([mol_num, min_num]), 1 / 3) * 0.8
        self.gindexes = (self.span / mgrid).round().astype(int)
        self.ggrids = self.span / self.gindexes
        indexes = [range(x) for x in self.gindexes]
        nodes = list(itertools.product(*indexes))
        self.graph.add_nodes_from(nodes)
        for node in nodes:
            for ids in self.INIT_NBR_INCR:
                neighbor = tuple([
                    (x + y) % z for x, y, z in zip(node, ids, self.gindexes)
                ])
                self.graph.add_edge(neighbor, node)

    def rmClashNodes(self):
        xyzs = self.frm.loc[list(self.extg_gids)]
        nodes = (xyzs / self.ggrids).round().astype(int)
        nodes = set([tuple(x[1]) for x in nodes.iterrows()])
        rnodes = []
        for node in nodes:
            rnode = tuple([x % y for x, y in zip(node, self.gindexes)])
            rnodes.append(rnode)
        self.graph.remove_nodes_from(nodes)

    def getVoids(self, num=100):
        """
        Get the points from the voids.

        :param num int: number of voids returned
        :return list: list of points whether the void centers are
        """
        mcc = max(nx.connected_components(self.graph), key=len)
        cut = min(max(self.gindexes) / 3, (len(mcc) * 3 / 4 / np.pi)**(1 / 3))
        snum = num * 2 if len(mcc) > num * 2 else len(mcc)
        sampled = [x for x in random.sample(list(mcc), snum)]
        # yapf: disable
        largest_cc = {x: len(nx.single_source_shortest_path_length(self.graph, x, cutoff=cut))
                      for x in sampled}
        # yapf: enable
        largest_cc_rv = collections.defaultdict(list)
        for node, size in largest_cc.items():
            largest_cc_rv[size].append(node)
        sizes = sorted(set(largest_cc.values()), reverse=True)
        sel_nodes, sel_num = [], 0
        while len(sel_nodes) < num and sizes:
            size = sizes.pop(0)
            sub_nodes = largest_cc_rv[size]
            np.random.shuffle(sub_nodes)
            sel_nodes += sub_nodes
        return [
            self.ggrids * (np.random.normal(-0.5, 0.5, 3) + x)
            for x in sel_nodes
        ]

    def getDistsWithIds(self, ids):
        dists = [
            self.frm.getDists(list(self.extg_gids),
                              self.frm.loc[x]).reshape(-1, 1) for x in ids
        ]
        return np.concatenate(dists, axis=1)
