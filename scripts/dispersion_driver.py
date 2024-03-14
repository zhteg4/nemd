# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
Calculate dispersion by crystal build, symmetry search, displacement, force
constant, Kpace mesh and mode analysis.
"""
import os
import re
import sys
import glob
import subprocess

from nemd import xtal
from nemd import task
from nemd import jobutils
from nemd import logutils
from nemd import plotutils
from nemd import constants
from nemd import stillinger
from nemd import parserutils
from nemd import alamodeutils
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


class Dispersion(object):

    SUBMODULE_PATH = environutils.get_submodule_path()
    ALAMODE = environutils.ALAMODE
    ALAMODE_SRC = os.path.join(SUBMODULE_PATH, ALAMODE, ALAMODE)
    Si_LAMMPS = os.path.join(ALAMODE_SRC, 'example', 'Si_LAMMPS')
    SI_FF = os.path.join(Si_LAMMPS, 'Si.sw')
    LAMMPS_EXT = '.lammps'
    DFSET_HARMONIC_EXT = '.dfset_harmonic'
    PNG_EXT = '.png'

    def __init__(self, options):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.options = options
        self.xbuild = None
        self.lmp_dat = None
        self.orig_lammps_data = None
        self.datafiles = None
        self.dumpfiles = []
        self.dump_kfile = f"{self.options.jobname}{self.DFSET_HARMONIC_EXT}"
        self.afcs_xml = None
        self.ph_bonds_file = None

    def run(self):
        self.buildCell()
        self.writeDataFile()
        self.writeDispPattern()
        self.writeDisplacement()
        self.writeForce()
        self.writeForceConstant()
        self.writeDispersion()
        self.plotDispersion()

    def buildCell(self):
        """
        Build the supercell based on the unit cell.
        """
        self.xbuild = xtal.CrystalBuilder(
            self.options.name,
            jobname=self.options.jobname,
            dim=self.options.dimension,
            scale_factor=self.options.scale_factor)
        self.xbuild.run()
        log(f"The supper cell is created with the lattice parameters "
            f"being {self.xbuild.scell.lattice_parameters}")

    def writeDispPattern(self):
        self.disp_pattern_file = self.xbuild.writeDispPattern()
        log(f"{alamodeutils.AlaLogReader.SUGGESTED_DSIP_FILE} {self.disp_pattern_file}"
            )

    def writeDataFile(self):
        mol = self.xbuild.getMol()
        tasks = [stillinger.LammpsData.XYZ, stillinger.LammpsData.FORCE]
        self.lmp_dat = stillinger.LammpsData({1: mol},
                                             self.SI_FF,
                                             self.options.jobname,
                                             tasks=tasks)
        self.lmp_dat.writeData()
        self.orig_lammps_data = self.lmp_dat.lammps_data
        log(f"LAMMPS data file written as {self.orig_lammps_data}")

    def writeDisplacement(self):
        cmd = f"{jobutils.RUN_NEMD} displace.py --LAMMPS {self.orig_lammps_data} " \
              f"--prefix {self.options.jobname} --mag 0.01 " \
              f"-pf {self.disp_pattern_file}"
        info = subprocess.run(cmd, capture_output=True, shell=True)
        if bool(info.stderr):
            raise ValueError(info.stderr)
        dsp_logfile = f'{self.options.jobname}_dsp.log'
        with open(dsp_logfile, 'wb') as fh:
            fh.write(info.stdout)
        name = self.lmp_dat.lammps_data[:-len(self.lmp_dat.DATA_EXT)]
        self.datafiles = glob.glob(f"{name}*{self.LAMMPS_EXT}")
        log(f"Data files with displacements are written as: {self.datafiles}")

    def writeForce(self):
        name = self.lmp_dat.lammps_data[:-len(self.lmp_dat.DATA_EXT)]
        pattern = f"{name}(.*){self.LAMMPS_EXT}"
        for datafile in self.datafiles:
            index = re.search(pattern, datafile).groups()[0]
            jobname = f"{self.options.jobname}{index}"
            self.lmp_dat.resetFilenames(jobname)
            self.lmp_dat.lammps_data = f"{jobname}{self.LAMMPS_EXT}"
            self.lmp_dat.writeLammpsIn()
            self.dumpfiles.append(self.lmp_dat.lammps_dump)
            lmp_log = f"{self.options.jobname}{index}{task.Lammps_Driver.DRIVER_LOG}"
            cmd = f"{task.Lammps_Driver.LMP_SERIAL} {task.Lammps_Driver.FLAG_IN} " \
                  f"{self.lmp_dat.lammps_in} {task.Lammps_Driver.FLAG_LOG} {lmp_log} " \
                  f"{task.Lammps_Driver.FLAG_SCREEN} none"
            log(f"Running {cmd}")
            subprocess.run(cmd, capture_output=True, shell=True)

    def writeForceConstant(self):
        cmd = f"{jobutils.RUN_NEMD} extract.py --LAMMPS {self.orig_lammps_data} " \
              f"{' '.join(self.dumpfiles)}"
        info = subprocess.run(cmd, capture_output=True, shell=True)
        with open(self.dump_kfile, 'wb') as fh:
            fh.write(info.stdout)
        self.afcs_xml = self.xbuild.writeForceConstant()
        log(f"{alamodeutils.AlaLogReader.INPUT_FOR_ANPHON} {self.afcs_xml} ")

    def writeDispersion(self):
        self.ph_bonds_file = self.xbuild.writePhbands()
        log(f"{alamodeutils.AlaLogReader.PHONON_BAND_STRUCTURE} is saved as "
            f"{self.ph_bonds_file}")

    def plotDispersion(self):
        plotter = plotutils.DispersionPlotter(self.ph_bonds_file)
        plotter.run()
        fname = self.options.jobname + self.PNG_EXT
        plotter.fig.savefig(fname)
        jobutils.add_outfile(fname, jobname=self.options.jobname)
        log(f'Figure of phonon dispersion saved as {fname}')


def get_parser(parser=None, jflags=None):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :param jflags list: specific job control related flags to add
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
    parserutils.add_job_arguments(parser,
                                  arg_flags=jflags,
                                  jobname=environutils.get_jobname(JOBNAME))
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
    logger = logutils.createDriverLogger(jobname=options.jobname,
                                         log_file=True)
    logutils.logOptions(logger, options)
    dispersio = Dispersion(options)
    dispersio.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
