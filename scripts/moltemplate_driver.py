import math
import sys
import argparse
import logutils
import os
import units
import parserutils
import fileutils
import nemd
import plotutils
import environutils
import jobutils
import numpy as np
from rdkit import Chem

FlAG_CRU = 'cru'
FlAG_CRU_NUM = '-cru_num'
FlAG_BOND_LEN = '-bond_len'
FLAG_BOND_ANG = '-bond_ang'

MOLT_OUT_EXT = '.lt'

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')

LEN_CH = 1.0930  # length of the C-H bond
# ~= 109.5 degrees = tetrahedronal angle (C-C-C angle)
TETRAHEDRONAL_ANGLE = 2 * math.atan(math.sqrt(2))


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
    parser.add_argument(FlAG_CRU,
                        metavar=FlAG_CRU.upper(),
                        type=parserutils.type_monomer_smiles,
                        help='')
    parser.add_argument(FlAG_CRU_NUM,
                        metavar=FlAG_CRU_NUM[1:].upper(),
                        type=parserutils.type_positive_int,
                        help='')

    parser.add_argument(
        FlAG_BOND_LEN,
        metavar=FlAG_BOND_LEN[1:].upper(),
        type=parserutils.type_positive_float,
        default=1.5350,  # length of the C-H bond
        help='')
    parser.add_argument(
        FLAG_BOND_ANG,
        metavar=FLAG_BOND_ANG[1:].upper(),
        type=parserutils.type_positive_float,
        default=109.5,  # Tetrahedronal angle (C-C-C angle)
        help='')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


class Polymer(object):

    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.outfile = self.jobname + MOLT_OUT_EXT

    def run(self):
        self.setBondProj()
        log('Finished', timestamp=True)

    def setBondProj(self):
        bond_ang = self.options.bond_ang / 2. / 180. * math.pi
        bond_proj = self.options.bond_len * math.sin(bond_ang)
        pass


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    polm = Polymer(options, jobname)
    polm.run()


if __name__ == "__main__":
    main(sys.argv[1:])
