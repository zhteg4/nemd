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

FLAG_IN_FILE = 'in_file'
FLAG_TEMP_FILE = '-temp_file'
FlAG_ENEGER_FILE = '-energy_file'
FlAG_LOG_FILE = '-log_file'
FLAG_CROSS_SECTIONAL_AREA = '-cross_sectional_area'

LOG_LAMMPS = 'log.lammps'

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


def log_debug(msg):
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def get_parser():
    parser = parserutils.get_parser(
        description=
        'Calculate thermal conductivity using non-equilibrium molecular dynamics.'
    )
    parser.add_argument(FLAG_IN_FILE,
                        metavar=FLAG_IN_FILE.upper(),
                        type=parserutils.type_file,
                        help='')
    parser.add_argument(FlAG_LOG_FILE,
                        metavar=FlAG_LOG_FILE.upper(),
                        type=parserutils.type_file,
                        help='')
    parser.add_argument(FLAG_TEMP_FILE,
                        metavar=FLAG_TEMP_FILE.upper(),
                        type=parserutils.type_file,
                        help='')
    parser.add_argument(FlAG_ENEGER_FILE,
                        metavar=FlAG_ENEGER_FILE.upper(),
                        type=parserutils.type_file,
                        help='')
    parser.add_argument(FLAG_CROSS_SECTIONAL_AREA,
                        metavar='ANGSTROM^2',
                        type=parserutils.type_positive_float,
                        help='')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)

    if options.log_file is None:
        in_file_dir = os.path.dirname(options.in_file)
        log_file = os.path.join(in_file_dir, LOG_LAMMPS)
        try:
            options.log_file = parserutils.type_file(log_file)
        except argparse.ArgumentTypeError:
            parser.error(f'{log_file} not found. ({FlAG_LOG_FILE})')

    if options.temp_file and options.energy_file:
        return options

    lammps_in = fileutils.LammpsInput(options.in_file)
    lammps_in.run()

    if options.temp_file is None:
        temp_file = lammps_in.getTempFile()
        if temp_file is None:
            parser.error(
                f"{options.in_file} doesn't define a temperature file. ({FLAG_TEMP_FILE})"
            )
        try:
            options.temp_file = parserutils.type_file(temp_file)
        except argparse.ArgumentTypeError:
            parser.error(
                f'{temp_file} from {options.in_file} not found. ({FLAG_TEMP_FILE})'
            )

    if options.energy_file is None:
        energy_file = lammps_in.getEnergyFile()
        if energy_file is None:
            parser.error(
                f"{options.in_file} doesn't define a energy file. ({FlAG_ENEGER_FILE})"
            )
        try:
            options.energy_file = parserutils.type_file(energy_file)
        except argparse.ArgumentTypeError:
            parser.error(
                f'{energy_file} from {options.in_file} not found. ({FlAG_ENEGER_FILE})'
            )

    return options


class Nemd(object):
    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.lammps_in = None
        self.lammps_temp = None
        self.lammps_energy = None
        self.timestep = None

    def run(self):
        self.loadLammpsIn()
        self.loadLog()
        self.loadTemp()
        self.loadEne()
        self.plot()
        self.setThermalConductivity()
        log('Finished', timestamp=True)

    def loadLammpsIn(self):
        self.lammps_in = fileutils.LammpsInput(self.options.in_file)
        self.lammps_in.run()

        self.lammps_units = self.lammps_in.getUnits()
        log(f"Lammps units is {self.lammps_units}.")
        self.timestep = self.lammps_in.getTimestep()
        log(f"Timestep is {self.timestep} fs.")

    def loadLog(self):
        self.lammps_log = fileutils.LammpsLogReader(self.options.log_file, self.options.cross_sectional_area)
        self.lammps_log.run()
        log(f"The cross sectional area is {self.lammps_log.cross_sectional_area:0.4f} Angstroms^2"
            )

    def loadTemp(self):
        block_num = 5
        self.lammps_temp = fileutils.TempReader(self.options.temp_file,
                                                block_num=block_num)
        self.lammps_temp.run()
        self.lammps_temp.write(self.jobname + '_temp')
        log(f"Every {int(self.lammps_temp.frame_num / block_num)} successive temperature profiles out of "
            f"{self.lammps_temp.frame_num} are block-averaged")

    def loadEne(self):
        self.lammps_energy = fileutils.EnergyReader(self.options.energy_file,
                                                    self.timestep)
        self.lammps_energy.run()
        self.lammps_energy.write(self.jobname + '_ene')
        log(f"Found {self.lammps_energy.total_step_num} steps of energy logging, "
            f"corresponding to {self.lammps_energy.total_step_num * self.timestep / units.NANO2FETO} ns"
            )

    def plot(self):
        temp_ene_plotter = plotutils.TempEnePlotter(self.lammps_temp,
                                                    self.lammps_energy,
                                                    self.jobname)
        temp_ene_plotter.plot()

    def setThermalConductivity(self):
        thermal_conductivity = nemd.ThermalConductivity(
            self.lammps_temp.slope,
            self.lammps_energy.slope,
            self.lammps_log.cross_sectional_area,
        )
        thermal_conductivity.run()
        log(f"Thermal conductivity is {thermal_conductivity.thermal_conductivity:.4f} W / (m * K)"
            )


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    nemd = Nemd(options, jobname)
    nemd.run()


if __name__ == "__main__":
    main(sys.argv[1:])
