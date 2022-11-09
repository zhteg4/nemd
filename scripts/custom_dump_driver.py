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

    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
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
        #self.checkFrames()
        self.write()
        log('Finished', timestamp=True)

    def setStruct(self):
        if not self.options.data_file:
            return

        self.data_reader = oplsua.DataFileReader(self.options.data_file)
        self.data_reader.run()

    def checkFrames(self, threshold=2.):

        for idx, frm in enumerate(self.getFrames()):
            clashes = self.getClashes(frm, threshold=threshold)
            if not clashes:
                continue
            log(f"Frame {idx} has {len(clashes)} clashes.")
        log('All frames are checked for clashes.')

    def getClashes(self, frm, threshold=2.0):
        clashes = []
        nids, cell, grids, indexes, span = self.getNeighborCell(frm,
                                                                cut=10,
                                                                resolution=2)
        span = np.array(span)

        for idx, row in frm.iterrows():
            neighbors = self.getNeighbors(row.values, nids, cell, grids,
                                          indexes)
            for neighbor in neighbors:
                nxyz = frm.loc[neighbor]
                deta = row.values - nxyz
                ndetal = round(deta / span)
                dist = math.dist((
                    0,
                    0,
                    0,
                ), deta - ndetal * span)
                if dist <= threshold:
                    clashes.append((idx, neighbor))
        return clashes

    def getNeighbors(self, xyz, nids, cell, grids, indexes):

        id = (xyz / grids).round().astype(int)
        ids = [(id + nid) % indexes for nid in nids]
        return [y for x in ids for y in cell[tuple(x)]]

    def getNeighborCell(self, frm, cut=10, resolution=2):
        box = frm.attrs['box']
        span = [box[i * 2 + 1] - box[i * 2] for i in range(3)]
        indexes = [math.ceil(x / resolution) for x in span]
        grids = [x / i for x, i in zip(span, indexes)]

        cut_mids = [math.ceil(cut / x) for x in grids]
        grids = np.array(grids)
        neigh_ids = [
            ijk for ijk in itertools.product(
                *[range(cut_mids[x]) for x in range(3)])
            if math.dist((0, 0, 0), grids * ijk) <= cut
        ]
        neigh_ids.remove((
            0,
            0,
            0,
        ))
        all_neigh_ids = set([
            tuple(np.array(ijk) * signs)
            for signs in itertools.product((-1, 1), (-1, 1), (-1, 1))
            for ijk in neigh_ids
        ])

        ids = ((frm) / grids).round().astype(int)
        atom_cell = collections.defaultdict(list)
        for idx, row in ids.iterrows():
            atom_cell[(
                row.xu,
                row.yu,
                row.zu,
            )].append(idx)
        return all_neigh_ids, atom_cell, grids, indexes, span

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

    def write(self, wrapped=True, bond_across_pbc=False):
        with open(self.outfile, 'w') as self.out_fh:
            for frm in self.getFrames():
                if wrapped:
                    box = frm.attrs['box']
                    span = np.array(
                        [box[i * 2 + 1] - box[i * 2] for i in range(3)])
                    if bond_across_pbc:
                        frm = frm % span
                    else:
                        for mol in self.data_reader.mols.values():
                            center = frm.loc[mol].mean()
                            delta = (center % span) - center
                            frm.loc[mol] += delta
                self.out_fh.write(f'{frm.shape[0]}\n')
                index = [self.data_reader.atoms[x].ele for x in frm.index]
                frm.index = index
                frm.to_csv(self.out_fh,
                           mode='a',
                           index=True,
                           sep=' ',
                           header=True)


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
