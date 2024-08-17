# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver to build crystal.
"""
import os
import sys

from nemd import xtal
from nemd import jobutils
from nemd import logutils
from nemd import constants
from nemd import stillinger
from nemd import parserutils
from nemd import environutils

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

FlAG_NAME = '-name'
FlAG_DIMENSION = '-dimension'
FlAG_DELTA = '-delta'
FlAG_NAME = '-name'
FLAG_SCALED_FACTOR = '-scale_factor'


def log_debug(msg):
    """
    Print this message into the log file in debug mode.

    :param msg str: the msg to be printed
    """
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: print time after the msg
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


class CrystalBuilder(object):

    SUBMODULE_PATH = environutils.get_submodule_path()
    ALAMODE = environutils.ALAMODE
    ALAMODE_SRC = os.path.join(SUBMODULE_PATH, ALAMODE, ALAMODE)
    Si_LAMMPS = os.path.join(ALAMODE_SRC, 'example', 'Si_LAMMPS')
    SI_FF = os.path.join(Si_LAMMPS, 'Si.sw')

    def __init__(self, options):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.options = options

    def run(self):
        xbuild = xtal.CrystalBuilder(self.options.name,
                                     dim=self.options.dimension,
                                     scale_factor=self.options.scale_factor)
        xbuild.run()
        mol = xbuild.getMol()
        lmp_dat = stillinger.Struct.fromMols([mol],
                                             ff=self.SI_FF,
                                             options=self.options)
        lmp_dat.writeData()
        log(f"LAMMPS data file written as {lmp_dat.datafile}")
        lmp_dat.writeLammpsIn()
        log(f"LAMMPS input script written as {lmp_dat.inscript}")
        jobutils.add_outfile(lmp_dat.datafile, jobname=self.options.jobname)
        jobutils.add_outfile(lmp_dat.inscript,
                             jobname=self.options.jobname,
                             set_file=True)


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
    # crystals.Crystal.builtins
    parser.add_argument(
        FlAG_NAME,
        default='Si',
        metavar=FlAG_NAME.upper()[1:],
        choices=['Si'],
        help='Name to retrieve the crystal structure from the database.')
    parser.add_argument(
        FlAG_DIMENSION,
        default=constants.ONE_ONE_ONE,
        type=int,
        metavar=FlAG_DIMENSION.upper()[1:],
        nargs='+',
        help='Unit cell is duplicated along each lattice vector by the '
        'corresponding factor.')
    parser.add_argument(
        FLAG_SCALED_FACTOR,
        nargs='+',
        default=constants.ONE_ONE_ONE,
        metavar=FLAG_SCALED_FACTOR.upper()[1:],
        type=parserutils.type_positive_float,
        help='Each lattice vector is scaled by the cor factor.')
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    if len(options.scale_factor) == 2:
        parser.error(f"{FLAG_SCALED_FACTOR} is three floats for each lattice "
                     f"vector. ({options.scale_factor} found)")
    options.scale_factor = options.scale_factor[:3]
    if len(options.scale_factor) == 1:
        options.scale_factor = options.scale_factor * 3
    return options


logger = None


def main(argv):

    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=JOBNAME)
    logutils.logOptions(logger, options)
    xtal_builder = CrystalBuilder(options)
    xtal_builder.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
