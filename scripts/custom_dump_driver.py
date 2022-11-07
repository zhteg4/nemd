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
        self.write()
        log('Finished', timestamp=True)

    def read(self):
        with open(self.options.custom_dump, 'r') as self.dmp_fh:
            while True:
                lines = [self.dmp_fh.readline() for _ in range(9)]
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
                if frm.shape[0] != atom_num:
                    break
                frm.attrs['box'] = box
                yield frm

    def write(self):
        with open(self.outfile, 'w') as self.out_fh:
            for frm in self.read():
                self.out_fh.write(f'{frm.shape[0]}\n')
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
