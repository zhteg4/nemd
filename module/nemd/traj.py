# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module read, parser, and analyze trajectories.

Unzip a GZ trajectory file: gzip -dv dump.custom.gz
"""
import io
import os
import math
import gzip

import numba
import types
import base64
import warnings
import itertools
import functools
import subprocess
import collections
import numpy as np
import pandas as pd
import networkx as nx
from contextlib import contextmanager

from nemd import symbols
from nemd import lammpsdata
from nemd import numbautils
from nemd import environutils


def frame_steps(filename, marker='ITEM: TIMESTEP'):
    """
    Get the frame steps.

    :param filename str: the filename to read frames
    :param marker str: the marker to find the step information
    :return 'numpy.ndarray': the step information of all steps
    """
    info = subprocess.run(
        f"zgrep -A1 '{marker}' {filename} | sed '/{marker}/d;/^--$/d'",
        capture_output=True,
        shell=True)
    return np.loadtxt(io.StringIO(info.stdout.decode("utf-8")), dtype=int)


def slice_frames(filename=None, contents=None, slices=None, start=0):
    """
    Get and slice the trajectory frames.

    :param filename str: the filename to read frames
    :param contents `bytes`: parse the contents if filename not provided
    :param slices list: start, stop, and interval
    :param start int: only frames with step >= this value will be fully read
    :return iterator of 'Frame': each frame has coordinates and box info
    """
    frm_iter = get_frames(filename=filename, contents=contents, start=start)
    if not slices:
        return frm_iter
    return itertools.islice(frm_iter, *slices)


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


class XYZ(np.ndarray):
    """
    XYZ class to allow fast access to xyz coordinate by global atom ids.
    """

    def __new__(cls, frame, *args, **kwargs):
        """
        :param frame pandas.core.frame.DataFrame: the trajectory dataframe with
            the index being atom global ids and values being xyz coordinates.
        """
        obj = np.asarray(frame.values).view(cls)
        return obj


class Frame(pd.DataFrame):
    """
    Class to hold coordinate information.
    """

    XYZU = symbols.XYZU

    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = pd.DataFrame._internal_names + ['box', 'step', 'xyz']
    _internal_names_set = set(_internal_names)

    def __init__(self,
                 data=None,
                 box=None,
                 index=None,
                 columns=None,
                 step=None,
                 dtype=None,
                 **kwargs):
        """
        :param data nx3 'numpy.ndarray' or 'DataFrame': xyz data
        :param box str: xlo, xhi, ylo, yhi, zlo, zhi boundaries
        :param index list: the atom indexes
        :param columns list: the data columns (e.g., xu, yu, zu, element)
        :param step int: the number of simulation step that this frame is at
        :param dtype str: the data type of the frame
        """
        super().__init__(data=data, index=index, dtype=dtype, **kwargs)
        self.box = box
        self.step = step
        cols = self.XYZU if columns is None else columns
        self.rename(columns={i: x for i, x in enumerate(cols)}, inplace=True)
        self.xyz = XYZ(self)
        if not isinstance(data, Frame):
            return
        if self.box is None and data.box is not None:
            self.box = data.box.copy()
        if self.step is None:
            self.step = data.step

    @property
    def _constructor(self):
        """
        Return the constructor of the class.

        :return 'Frame': the constructor of the class
        """
        return Frame

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
                try:
                    lines = [fh.readline() for _ in range(9)]
                except EOFError:
                    return
                if not all(lines):
                    return
                atom_num = int(lines[3].rstrip())
                step = int(lines[1].rstrip())
                if step < start:
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
                    box = lammpsdata.Box.fromLines(lines[5:8])
                    # 'xu', 'yu', 'zu'
                    columns = lines[-1].rstrip().split()[-3:]
                    frame = cls(data=data[:, 1:],
                                box=box,
                                index=data[:, 0].astype(int),
                                columns=columns,
                                step=step)
                    frame.index -= 1
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

    @property
    def volume(self):
        """
        Get the volume of the frame.

        :param float: the volume of the frame
        """
        return np.prod(self.box.span)

    def pairDists(self, grp1=None, grp2=None):
        """
        Get the distance between atom pair.

        :param grp1 list: list of gids as the atom selection
        :param grp2 list of list: each sublist contains atom ids to compute
            distances with one atom in grp1.
        return numpy.ndarray: list of distance between pairs.
        """
        grp1 = self.index if grp1 is None else sorted(grp1)
        grp2 = (grp1[:i] for i in range(len(grp1))) if grp2 is None else grp2
        dists = [self.getDists(x, self.xyz[y, :]) for x, y in zip(grp2, grp1)]
        return np.concatenate(dists) if dists else np.array(dists)

    def getDists(self, ids, xyz):
        """
        Get the distance between the xyz and the of the xyzs associated with the
        input atom ids.

        :param ids list of int: atom ids
        :param xyz (3,) 'pandas.core.series.Series': xyz coordinates and atom id
        :return list of floats: distances
        """
        dists = self.xyz[ids, :] - np.array(xyz)

        if environutils.get_python_mode() == environutils.ORIGINAL_MODE:
            for id in range(3):
                func = lambda x: math.remainder(x, self.box.span[id])
                dists[:, id] = np.frompyfunc(func, 1, 1)(dists[:, id])
            return np.linalg.norm(dists, axis=1)

        return np.array(self.remainderIEEE(dists, self.box.span))

    @staticmethod
    @numbautils.jit
    def remainderIEEE(dists, span):
        """
        Calculate IEEE 754 remainder.

        https://stackoverflow.com/questions/26671975/why-do-we-need-ieee-754-remainder

        :param dists numpy.ndarray: distances
        :param span numpy.ndarray: box span
        :return list of floats: distances within half box span
        """
        dists -= np.round(np.divide(dists, span)) * span
        return [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in dists]

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

        if broken_bonds:
            self.xyz = self.xyz % self.box.span
            # The wrapped xyz shouldn't support molecule center operation
            return

        # The unwrapped xyz can directly perform molecule center operation
        for gids in dreader.molecules.values():
            center = self.xyz[gids, :].mean(axis=0)
            delta = (center % self.box.span) - center
            self.xyz[gids, :] += delta

    def glue(self, dreader=None):
        """
        Circular mean to compact the molecules. Good for molecules droplets in
        vacuum. (extension needed for droplets or clustering in solution)

        :param dreader 'oplsua.DataFileReader': to get molecule ids and
            associated atoms are available
        """
        if dreader is None:
            return

        centers = pd.concat(
            [self.loc[x].mean(axis=0) for x in dreader.molecules.values()],
            axis=1).transpose()
        centers.index = dreader.molecules.keys()
        theta = centers / self.box.span * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta = np.arctan2(sin_theta.mean(), cos_theta.mean())
        mcenters = theta * self.box.span / 2 / np.pi
        cshifts = (
            (mcenters - centers) / self.box.span).round() * self.box.span
        for id, mol in dreader.molecules.items():
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
            type_ids = dreader.atoms.type_id.loc[data.index]
            data.index = dreader.masses.element[type_ids]
        header = self.box.to_str()
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
                    float_format='%.4f')

    def assign(self, **kwargs):
        """
        Assign new columns to a DataFrame.

        :return: A new DataFrame with the new columns in addition.
        :rtype: DataFrame
        """
        df = super().assign(**kwargs)
        if self.box is not None:
            df.box = self.box.copy()
        df.step = self.step
        return df

    def update(self, other, **kwargs):
        """
        Update the frame.

        :param other: DataFrame, or object coercible into a DataFrame
        :type other: other
        """
        super().update(other, **kwargs)
        if isinstance(other, Frame):
            if self.box is not None:
                self.box = other.box.copy()
            self.step = other.step


class DistanceCell(Frame):
    """
    Class to quick search neighbors based on distance criteria and perform clash
    checking.
    """

    GRID_MAX = 20
    # https://pandas.pydata.org/docs/development/extending.html
    _internal_names = Frame._internal_names + [
        'cut', 'nbr_ids', 'atom_cell', 'graph', 'orig_values',
        'orig_atom_cell', 'span', 'grids', 'indexes', 'indexes_numba',
        'nbr_map', 'gindexes', 'ggrids', 'orig_graph', 'grids', 'radii',
        'excluded', 'gids', 'orig_gids'
    ]
    _internal_names_set = set(_internal_names)

    def __init__(self, data=None, gids=None, cut=None, struct=None):
        """
        :param data 'Frame': trajectory frame
        :param gids list: global atom ids to analyze
        :param cut float: the cutoff distance to search neighbors
        :param struct 'Struct' or 'DataFileReader': radii and excluded pairs
            are set from this object.
        """
        super().__init__(data=data)
        self.cut = cut
        self.gids = gids
        self.struct = struct
        self.radii = None
        self.radii_numba = None
        self.nbr_ids = None
        self.atom_cell = None
        self.graph = None
        self.vals = None
        self.cell_vals = None
        self.indexes = None
        self.indexes_numba = None
        self.grids = None
        self.excluded = None
        self.excluded_numba = None
        if self.gids is None:
            self.gids = set(range(self.shape[0]))
        self.orig_gids = self.gids.copy()
        self.orig_values = self.values.copy()

    def setUp(self):
        """
        Set up the distance cell.
        """
        self.setRadius()
        self.setCut()
        self.setExcluded()
        self.setExcludedNumba()
        self.setGrids()
        self.setNeighborIds()
        self.setNeighborMap()
        self.setAtomCell()

    @functools.singledispatchmethod
    def setup(self, arg):
        """
        Set up the distance cell with additional arguments.
        """
        raise NotImplementedError("Cannot set up the distance cell.")

    @setup.register
    def _(self, arg: Frame):
        """
        Set up the distance cell from input trajectory frame.

        :param arg: the input trajectory frame.
        :type arg: `Frame`
        """
        self[self.XYZU] = arg
        self.orig_values = self.values.copy()
        self.xyz = XYZ(self)
        self.step = arg.step
        self.setup(arg.box)

    @setup.register
    def _(self, arg: lammpsdata.Box):
        """
        Set up the distance cell from input periodic boundary conditions.

        :param arg: the input box.
        :type arg: `Box`
        """
        self.box = arg
        self.setUp()

    def setRadius(self):
        """
        Set the vdw radius.
        """
        if self.radii is not None:
            return

        if self.struct:
            atoms = self.struct.atoms
            pair_coeffs = self.struct.pair_coeffs
            self.radii = lammpsdata.Radius(pair_coeffs.dist, atoms.type_id)
            self.radii_numba = np.array(self.radii).astype(float)
            return

        atoms = lammpsdata.Atom(self.shape[0])
        pair_coeffs = lammpsdata.PairCoeff(self.shape[0])
        pair_coeffs.dist = lammpsdata.Radius.MIN_DIST
        self.radii = lammpsdata.Radius(pair_coeffs.dist, atoms.type_id)
        self.radii_numba = np.array(self.radii).astype(float)

    def setCut(self):
        """
        Set the cut-off distance.
        """
        if self.cut is not None:
            return
        self.cut = self.radii.max()

    def setExcluded(self, include14=True):
        """
        Set the pair exclusion during clash check. Bonded atoms and atoms in
        angles are in the exclusion. The dihedral angles are in the exclusion
        if include14=True.

        :param include14 bool: If True, 1-4 interaction in a dihedral angle count
            as exclusion.
        """
        if self.excluded is not None:
            return

        self.excluded = collections.defaultdict(set)
        for idx in self.index:
            self.excluded[idx].add(idx)

        if self.struct is None:
            return

        pairs = set(self.struct.bonds.getPairs())
        pairs = pairs.union(self.struct.angles.getPairs())
        pairs = pairs.union(self.struct.impropers.getPairs())
        if include14:
            pairs = pairs.union(self.struct.dihedrals.getPairs())
        for id1, id2 in pairs:
            self.excluded[id1].add(id2)
            self.excluded[id2].add(id1)

    def setExcludedNumba(self):
        """
        Set the pair exclusion during clash check.
        """
        self.excluded_numba = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64[:],
        )

        for key, val in self.excluded.items():
            self.excluded_numba[key] = np.array(list(val)).astype(np.int64)

    def setGrids(self, max_num=GRID_MAX):
        """
        Set grids and indexes.

        :param max_num int: maximum number of cells in each dimension.
        Indexes: the number of cells in three dimensions
        Grids: the length of the cell in each dimension
        """
        self.indexes = [math.ceil(x / self.cut) for x in self.box.span]
        self.indexes = [min(x, max_num) for x in self.indexes]
        self.indexes_numba = numba.int32(self.indexes)
        self.grids = np.array(
            [x / i for x, i in zip(self.box.span, self.indexes)])

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
        nbr_ids = [x for x in ijks if separation_dist(x) < self.cut]
        # Keep itself (0,0,0) cell as multiple atoms may be in one cell.
        signs = itertools.product((-1, 1), (-1, 1), (-1, 1))
        signs = [np.array(x) for x in signs]
        uq_nbr_ids = set([tuple(y * x) for x in signs for y in nbr_ids])
        self.nbr_ids = np.array(list(uq_nbr_ids))

    def setNeighborMap(self):
        """
        Set map between node id to neighbor node ids.
        """
        if environutils.get_python_mode() != environutils.ORIGINAL_MODE:
            self.nbr_map = self.getNbrMap(self.indexes_numba, self.nbr_ids)
            return

        nbr_map = np.zeros((*self.indexes, *self.nbr_ids.shape), dtype=int)
        nodes = list(itertools.product(*[range(x) for x in self.indexes]))
        for node in nodes:
            nbr_map[node] = (self.nbr_ids + node) % self.indexes
        cols = list(itertools.product(*[range(x) for x in self.indexes]))
        unique_maps = [np.unique(nbr_map[tuple(x)], axis=0) for x in cols]
        shape = np.unique([x.shape for x in unique_maps], axis=0).max(axis=0)
        self.nbr_map = np.zeros((*self.indexes, *shape), dtype=int)
        for col, unique_map in zip(cols, unique_maps):
            self.nbr_map[col[0], col[1], col[2], :, :] = unique_map
        # getNbrMap() and the original mode generate nbr_map in different
        # order: np.unique(nbr_map[i, j, j, :, :], axis=0) remains the same

    @staticmethod
    @numbautils.jit(parallel=True)
    def getNbrMap(indexes, nbr_ids, nopython):
        """
        Get map between node id to neighbor node ids.

        :param indexes numpy.ndarray: the number of cells in three dimensions
        :param nbr_ids numpy.ndarray: Neighbors cells (separation distances
            less than the cutoff)
        :param nopython bool: whether numba nopython mode is on
        :return numpy.ndarray: map between node id to neighbor node ids
        """
        # Unique neighbor cell ids
        min_id = np.min(nbr_ids)
        shifted_nbr_ids = nbr_ids - min_id
        wrapped_nbr_ids = shifted_nbr_ids % indexes
        ushape = np.max(wrapped_nbr_ids) + 1
        boolean = numba.boolean if nopython else np.bool_
        uids = np.zeros((ushape, ushape, ushape), dtype=boolean)
        for wrapped_ids in wrapped_nbr_ids:
            uids[wrapped_ids[0], wrapped_ids[1], wrapped_ids[2]] = True
        uq_ids = np.array(list([list(x) for x in uids.nonzero()])).T + min_id
        # Build neighbor map based on unique neighbor ids
        shape = (indexes[0], indexes[1], indexes[2], len(uq_ids), 3)
        neigh_mp = np.empty(shape, dtype=numba.int32 if nopython else np.int32)
        for xid in numba.prange(indexes[0]):
            for yid in numba.prange(indexes[1]):
                for zid in numba.prange(indexes[2]):
                    idx = np.array([xid, yid, zid])
                    neigh_mp[xid, yid, zid, :, :] = (uq_ids + idx) % indexes
        return neigh_mp

    def setAtomCell(self):
        """
        Put atom ids into the corresponding cells.

        self.atom_cell.shape = [X index, Y index, Z index, all atom ids]
        """
        if environutils.get_python_mode() == environutils.ORIGINAL_MODE:
            ids = (self / self.grids).round().astype(int) % self.indexes
            self.atom_cell = np.zeros((*self.indexes, ids.shape[0] + 1),
                                      dtype=bool)
            for row in ids.loc[list(self.gids)].itertuples():
                self.atom_cell[row.xu, row.yu, row.zu][row.Index] = True
            self.orig_atom_cell = self.atom_cell.copy()
            return

        gids = numba.int32(list(self.gids))
        self.atom_cell = self.setAtomCellNumba(gids, self.values, self.grids,
                                               self.indexes_numba)
        self.orig_atom_cell = self.atom_cell.copy()

    @staticmethod
    @numbautils.jit
    def setAtomCellNumba(atom_ids, xyzs, grids, indexes, nopython):
        """
        Put atom ids into the corresponding cells.

        :param atom_ids 'numpy.ndarray': the atom gids
        :param xyzs 'numpy.ndarray': xyz of atom coordinates
        :param grids 'numpy.ndarray': the length of the cell in each dimension
        :param indexes list of numba.int32: the number of the cell in each dimension
        :param nopython bool: whether numba nopython mode is on
        :return 'numpy.ndarray': map between cell id to atom ids
            [X index, Y index, Z index, all atom ids]
        """
        int32 = numba.int32 if nopython else np.int32
        cids = np.round(xyzs / grids).astype(int32) % indexes
        shape = (indexes[0], indexes[1], indexes[2], cids.shape[0] + 1)
        boolean = numba.boolean if nopython else np.bool_
        atom_cell = np.zeros(shape, dtype=boolean)
        for aid, cid in zip(atom_ids, cids):
            atom_cell[cid[0], cid[1], cid[2]][aid] = True
        return atom_cell

    def reset(self):
        """
        Reset the distance cell.
        """
        self.gids = self.orig_gids.copy()
        self.iloc[:] = self.orig_values.copy()
        self.atom_cell[:] = self.orig_atom_cell.copy()
        if self.graph is not None:
            self.graph = self.orig_graph.copy()

    def add(self, gids):
        """
        Add gids to atom cell and existing gids.

        :param gids: the global atom ids to be added.
        :type gids: list
        """
        self.gids.update(gids)
        ids = (self.xyz[gids, :] / self.grids).round().astype(int)
        for idx, (ix, iy, iz) in zip(gids, ids % self.indexes):
            self.atom_cell[ix, iy, iz][idx] = True

    def remove(self, gids):
        """
        Remove gids from atom cell and existing gids.

        :param gids: the global atom ids to be removed.
        :type gids: list
        """
        self.gids = self.gids.difference(gids)
        ids = (self.xyz[gids, :] / self.grids).round().astype(int)
        for idx, (ix, iy, iz) in zip(gids, ids % self.indexes):
            self.atom_cell[ix, iy, iz][idx] = False

    def getNeighbors(self, xyz):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself)

        :param xyz 1x3 array of floats: xyz of one atom coordinates
        :return list int: the atom ids of the neighbor atoms
        """
        if environutils.get_python_mode() == environutils.ORIGINAL_MODE:
            id = (xyz / self.grids).round().astype(int) % self.indexes
            ids = self.nbr_map[tuple(id)]
            return [
                y for x in ids for y in self.atom_cell[tuple(x)].nonzero()[0]
            ]
        return self.getNeighborsNumba(xyz, self.grids, self.indexes_numba,
                                      self.nbr_map, self.atom_cell)

    @staticmethod
    @numbautils.jit
    def getNeighborsNumba(xyz, grids, indexes, nbr_map, atom_cell, nopython):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself) via Numba.

        :param xyz 1x3 'numpy.ndarray': xyz of one atom coordinates
        :param grids 'numpy.ndarray': the length of the cell in each dimension
        :param indexes list of 'numba.int32': the number of the cell in each dimension
        :param nbr_map ixjxkxnx3 'numpy.ndarray': map between cell id to neighbor cell ids
        :param atom_cell ixjxkxn array of floats: map cell id into containing atom ids
        :param nopython bool: whether numba nopython mode is on
        :return list of int: the atom ids of the neighbor atoms
        """
        # The cell id for xyz
        int32 = numba.int32 if nopython else np.int32
        idx = np.round(xyz / grids).astype(int32) % indexes
        ids = nbr_map[idx[0], idx[1], idx[2], :]
        # The atom ids from all neighbor cells
        neighbors = [
            y for x in ids for y in atom_cell[x[0], x[1], x[2], :].nonzero()[0]
        ]
        return neighbors

    @staticmethod
    @numbautils.jit
    def hasClashesNumba(gids, xyz, id_map, radii, excluded, grids, indexes,
                        nbr_map, cell, span, nopython):
        """
        Get the neighbor atom ids from the neighbor cells (including the current
        cell itself) via Numba.

        :param xyz nx3 'numpy.ndarray': global atom ids for selection
        :param id_map 1xn 'numpy.ndarray': map global atom ids to atom types
        :param radii nxn 'numpy.ndarray': the radius of atom type pairs
        :param excluded dict of int list: the atom ids to be excluded in clash check
        :param grids 'numpy.ndarray': the length of the cell in each dimension
        :param indexes list of 'numba.int32': the number of the cell in each dimension
        :param nbr_map ixjxkxnx3 'numpy.ndarray': map between cell id to neighbor cell ids
        :param cell ixjxkxn array of floats: map cell id into containing atom ids
        :param span 1x3 'numpy.ndarray': the span of the box
        :param nopython bool: whether numba nopython mode is on
        :return bool: whether the selected atoms have clashes
        """

        int32 = numba.int32 if nopython else np.int32
        idxs = np.round(xyz[gids, :] / grids).astype(int32) % indexes

        for gid, idx in zip(gids, idxs):
            ids = nbr_map[idx[0], idx[1], idx[2], :]
            nbrs = np.array([
                y for x in ids for y in cell[x[0], x[1], x[2], :].nonzero()[0]
                if y not in excluded[gid]
            ])
            if not nbrs.size:
                continue
            delta = xyz[nbrs, :] - xyz[gid, :]
            delta -= np.round(np.divide(delta, span)) * span
            dists = [np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) for x in delta]
            thresholds = radii[id_map[gid], id_map[nbrs]]
            if (np.array(dists) < thresholds).any():
                return True
        return False

    def hasClashes(self, gids=None):
        """
        Whether the selected atoms have clashes

        :param gids: global atom ids for atom selection.
        :type gids: set

        :return: whether the selected atoms have clashes
        :rtype: bool
        """
        if gids is None:
            gids = self.gids

        if environutils.get_python_mode() == environutils.ORIGINAL_MODE:
            dists = (self.getClash(x) for x in gids)
            try:
                next(itertools.chain.from_iterable(dists))
            except StopIteration:
                return False
            return True

        return self.hasClashesNumba(gids, self.xyz, self.radii.id_map,
                                    self.radii_numba, self.excluded_numba,
                                    self.grids, self.indexes_numba,
                                    self.nbr_map, self.atom_cell,
                                    self.box.span)

    def getClashes(self, gids=None):
        """
        Get the clashes distances.

        :param gids: global atom ids for atom selection.
        :type gids: set

        :return: the clash distances
        :rtype: list of float
        """
        if gids is None:
            gids = self.gids
        return [y for x in gids for y in self.getClash(x)]

    def getClash(self, gid):
        """
        Get the clashes between xyz and atoms in the frame.

        :param gid int: the global atom id
        :return list of float: clash distances between atom pairs
        """
        xyz = self.xyz[gid, :]
        neighbors = set(self.getNeighbors(xyz))
        neighbors = neighbors.difference(self.excluded[gid])
        if not neighbors:
            return []
        neighbors = list(neighbors)
        delta = self.xyz[neighbors, :] - xyz
        dists = np.array(self.remainderIEEE(delta, self.box.span))
        thresholds = self.radii[gid, neighbors]
        return dists[np.nonzero(dists < thresholds)]

    def setGraph(self, mol_num):
        """
        Set graph using grid intersection as nodes and connect neighbor nodes.
        """
        self.graph = nx.Graph()
        grid_num = math.ceil(pow(mol_num, 1 / 3)) + 1
        mgrid = self.box.span.min().min() / grid_num
        self.gindexes = (self.box.span / mgrid).round().astype(int)
        self.ggrids = self.box.span / self.gindexes
        indexes = [range(x) for x in self.gindexes]
        nodes = list(itertools.product(*indexes))
        self.graph.add_nodes_from(nodes)
        for node in nodes:
            for ids in self.getNbrIncr():
                neighbor = tuple([
                    (x + y) % z for x, y, z in zip(node, ids, self.gindexes)
                ])
                self.graph.add_edge(neighbor, node)
        self.orig_graph = self.graph.copy()

    @classmethod
    @functools.cache
    def getNbrIncr(cls, nth=1):
        first = math.ceil(nth / 3)
        second = math.ceil((nth - first) / 2)
        third = nth - first - second
        row = np.array([first, second, third])
        data = []
        for signs in itertools.product([-1, 1], [-1, 1], [-1, 1]):
            rows = signs * np.array([x for x in itertools.permutations(row)])
            data.append(np.unique(rows, axis=0))
        return np.unique(np.concatenate(data), axis=0)

    def resetGraph(self):
        """
        Rest the graph to the original state.
        """
        self.graph = self.orig_graph.copy()

    def rmClashNodes(self):
        """
        Remove nodes occupied by existing atoms.
        """
        if not self.gids:
            return
        nodes = (self.xyz[list(self.gids), :] / self.ggrids).round()
        nodes = [tuple(x) for x in nodes.astype(int)]
        self.graph.remove_nodes_from(nodes)

    def getVoids(self):
        """
        Get the points from the voids.

        :return `numpy.ndarray`: each value is one random point from the voids.
        """
        return (y for x in self.getVoid() for y in x)

    def getVoid(self):
        """
        Get the points from the largest void.

        :return `numpy.ndarray`: each row is one random point from the void.
        """
        largest_component = max(nx.connected_components(self.graph), key=len)
        void = np.array(list(largest_component))
        void_max = void.max(axis=0)
        void_span = void_max - void.min(axis=0)
        infinite = (void_span + 1 == self.gindexes).any()
        if infinite:
            # The void tunnels the PBC and thus points are uniformly distributed
            yield self.ggrids * (np.random.normal(0, 0.5, void.shape) + void)
            return
        # The void is surrounded by atoms and thus the center is biased
        imap = np.zeros(void_max + 1, dtype=bool)
        imap[tuple(np.transpose(void))] = True
        center = void.mean(axis=0).astype(int)
        max_nth = np.abs(void - center).max(axis=0).sum()
        for nth in range(max_nth):
            nbrs = center + self.getNbrIncr(nth=nth)
            nbrs = nbrs[(nbrs <= void_max).all(axis=1).nonzero()]
            nbrs = nbrs[imap[tuple(np.transpose(nbrs))].nonzero()]
            np.random.shuffle(nbrs)
            yield self.ggrids * (np.random.normal(0, 0.5, nbrs.shape) + nbrs)

    @property
    def ratio(self):
        """
        The ratio of the existing atoms.

        :return: the ratio of the existing gids with respect to the total atoms.
        :rtype: str
        """
        return f'{len(self.gids)} / {self.shape[0]}'

    def pairDists(self, gids=None, nbrs=None):
        """
        Get the pair distances between existing atoms.

        :param gids: the center atom global atom ids.
        :type gids: list
        :param nbrs: the neighbor global atom ids.
        :type nbrs: list

        :return: the pair distances
        :rtype: 'numpy.ndarray'
        """
        grp1 = sorted(self.gids) if gids is None else sorted(gids)
        if nbrs is None:
            nbrs = [self.getNeighbors(x) for x in self.xyz[grp1, :]]
        else:
            nbrs = [nbrs for _ in range(len(grp1))]
        grp2 = [[z for z in y if z < x] for x, y in zip(grp1, nbrs)]
        return super().pairDists(grp1=grp1, grp2=grp2)
