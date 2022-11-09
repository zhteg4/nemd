import math
import copy
import sys
import argparse
import logutils
import functools
import os
import sys

import opls
import units
import pandas as pd
import parserutils
import fileutils
import nemd
import itertools
import plotutils
import collections
import environutils
import jobutils
import symbols
import numpy as np
import oplsua
from rdkit import Chem
from rdkit.Chem import AllChem

FlAG_CUSTOM_DUMP = 'custom_dump'
FlAG_DATA_FILE = '-data_file'
FlAG_TASK = '-task'
CLASH = 'clash'
XYZ = 'xyz'

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


def log_debug(msg):
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_error(msg):
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


def get_parser():
    parser = parserutils.get_parser(
        description='Generate the moltemplate input *.lt')
    parser.add_argument(FlAG_CUSTOM_DUMP,
                        metavar=FlAG_CUSTOM_DUMP.upper(),
                        type=parserutils.type_file,
                        help='')
    parser.add_argument(FlAG_DATA_FILE,
                        metavar=FlAG_DATA_FILE.upper(),
                        type=parserutils.type_file,
                        help='')
    parser.add_argument(FlAG_TASK,
                        choices=[XYZ, CLASH],
                        default=[XYZ],
                        nargs='+',
                        help='')

    jobutils.add_job_arguments(parser)
    return parser


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
        neigh_ids.remove((0, 0, 0,)) # yapf: disable
        self.neigh_ids = set([
            tuple(np.array(ijk) * signs)
            for signs in itertools.product((-1, 1), (-1, 1), (-1, 1))
            for ijk in neigh_ids
        ])

    def setAtomCell(self, ):
        ids = ((self.frm) / self.grids).round().astype(int)
        self.atom_cell = collections.defaultdict(list)
        for idx, row in ids.iterrows():
            self.atom_cell[(
                row.xu,
                row.yu,
                row.zu,
            )].append(idx)

    def getNeighbors(self, xyz):

        id = (xyz / self.grids).round().astype(int)
        ids = [tuple((id + x) % self.indexes) for x in self.neigh_ids]
        return [y for x in ids for y in self.atom_cell[x]]

    def getClashes(self, row, threshold=2., excluded=None):
        xyz = row.values
        neighbors = self.getNeighbors(xyz)
        if excluded:
            neighbors = [x for x in neighbors if x not in excluded[row.name]]
        dists = np.linalg.norm(
            (self.frm.loc[neighbors] - xyz + self.hspan) % self.span -
            self.hspan,
            axis=1)
        clashes = [(row.name, x) for x, y in zip(neighbors, dists)
                   if y <= threshold]
        if clashes:
            import pdb
            pdb.set_trace()
        return clashes


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)

    if CLASH in options.task and not options.data_file:
        parser.error(
            f'Please specify {FlAG_DATA_FILE} to run {FlAG_TASK} {CLASH}')
    return options


class CustomDump(object):

    XYZ_EXT = '.xyz'

    def __init__(self, options, jobname, diffusion=False):
        self.options = options
        self.jobname = jobname
        self.diffusion = diffusion
        self.outfile = self.jobname + self.XYZ_EXT

    def run(self):
        self.setStruct()
        self.checkClashes()
        self.writeXYZ()
        log('Finished', timestamp=True)

    def setStruct(self):
        if not self.options.data_file:
            return

        self.data_reader = oplsua.DataFileReader(self.options.data_file)
        self.data_reader.run()
        self.data_reader.setClashExclusion()

    def checkClashes(self, threshold=1.5):

        if CLASH not in self.options.task:
            return

        for idx, frm in enumerate(self.getFrames()):
            clashes = self.getClashes(frm, threshold=threshold)
            log(f"Frame {idx} has {len(clashes)} clashes.")
        log('All frames are checked for clashes.')

    def getClashes(self, frm, threshold=2.0):
        clashes = []
        dcell = DistanceCell(frm=frm, cut=10, resolution=2.)
        dcell.setUp()
        for _, row in frm.iterrows():
            clashes += dcell.getClashes(row,
                                        threshold=threshold,
                                        excluded=self.data_reader.excluded)
        return clashes

    def getFrames(self):
        with open(self.options.custom_dump, 'r') as self.dmp_fh:
            while True:
                lines = [self.dmp_fh.readline() for _ in range(9)]
                if not all(lines):
                    return
                atom_num = int(lines[3].strip('\n'))
                box = np.array([
                    float(y) for x in range(5, 8)
                    for y in lines[x].strip('\n').split()
                ])
                names = lines[-1].strip('\n').split()[-4:]
                frm = pd.read_csv(self.dmp_fh,
                                  nrows=atom_num,
                                  header=None,
                                  delimiter='\s',
                                  index_col=0,
                                  names=names,
                                  engine='python')
                if frm.shape[0] != atom_num or frm.isnull().values.any():
                    break
                frm.attrs['box'] = box
                yield frm

    def writeXYZ(self, wrapped=True, bond_across_pbc=False, glue=True):
        if glue and not (wrapped and bond_across_pbc is False):
            raise ValueError(f'Glue moves molecules together like droplets.')

        if XYZ not in self.options.task:
            return

        with open(self.outfile, 'w') as self.out_fh:
            for frm in self.getFrames():
                self.wrapCoords(frm,
                                wrapped=wrapped,
                                bond_across_pbc=bond_across_pbc,
                                glue=glue)
                self.out_fh.write(f'{frm.shape[0]}\n')
                index = [self.data_reader.atoms[x].ele for x in frm.index]
                frm.index = index
                frm.to_csv(self.out_fh,
                           mode='a',
                           index=True,
                           sep=' ',
                           header=True)
        log(f"Coordinates are written into {self.outfile}")

    def wrapCoords(self, frm, wrapped=True, bond_across_pbc=False, glue=True):
        if not wrapped:
            return

        box = frm.attrs['box']
        span = np.array([box[i * 2 + 1] - box[i * 2] for i in range(3)])
        if bond_across_pbc:
            frm = frm % span
            return

        for mol in self.data_reader.mols.values():
            center = frm.loc[mol].mean()
            delta = (center % span) - center
            frm.loc[mol] += delta

        if not glue:
            return

        centers = pd.concat(
            [frm.loc[x].mean() for x in self.data_reader.mols.values()],
            axis=1).transpose()

        theta = centers / span * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta = np.arctan2(sin_theta.mean(), cos_theta.mean())
        mcenters = theta * span / 2 / np.pi
        cshifts = ((mcenters - centers) / span).round()
        for id, mol in self.data_reader.mols.items():
            cshift = cshifts.loc[id - 1]
            frm.loc[mol] += cshift * span


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    cdump = CustomDump(options, jobname)
    cdump.run()


if __name__ == "__main__":
    main(sys.argv[1:])
