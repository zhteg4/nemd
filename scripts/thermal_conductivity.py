import sys
import logutils
import os
import units
import parserutils
import fileutils
import plotutils
import environutils
import jobutils

FLAG_IN_FILE = 'in_file'
FLAG_TEMP_FILE = 'temp_file'
FlAG_ENEGER_FILE = 'energy_file'


JOBNAME = os.path.basename(__file__).split('.')[0]


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
    parser.add_argument(FLAG_TEMP_FILE,
                        metavar=FLAG_TEMP_FILE.upper(),
                        help='')
    parser.add_argument(FlAG_ENEGER_FILE,
                        metavar=FlAG_ENEGER_FILE.upper(),
                        help='')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options

class Nemd(object):

    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.lammps_in = None
        self.temp_data = None
        self.lammps_energy = None

    def run(self):
        self.loadLammpsIn()
        self.loadTemp()
        self.loadEne()
        self.plot()

    def loadLammpsIn(self):
        self.lammps_in = fileutils.LammpsInput(self.options.in_file)
        self.lammps_in.run()

        self.lammps_units = self.lammps_in.cmd_items[fileutils.LammpsInput.UNITS]
        log(f"{self.lammps_units} is the units.")
        self.timestep = self.lammps_in.cmd_items[fileutils.LammpsInput.TIMESTEP]
        log(f"{self.timestep} is the timestep.")

    def loadTemp(self):
        block_num = 5
        self.temp_data, temp_dat_num = fileutils.load_temp(self.options.temp_file,
                                                     block_num=block_num)
        log(f"Every {int(temp_dat_num / block_num)} successive temperature profiles out of "
            f"{temp_dat_num} are block averaged")

    def loadEne(self):
        self.lammps_energy = fileutils.EnergyReader(self.options.energy_file, self.timestep)
        self.lammps_energy.run()
        log(f"Found {self.lammps_energy.total_step_num} steps of energy logging, "
            f"corresponding to {self.lammps_energy.total_step_num * self.timestep / units.NANO2FETO} ns"
            )

    def plot(self):
        temp_ene_plotter = plotutils.TempEnePlotter(self.temp_data, self.lammps_energy.data, self.jobname)
        temp_ene_plotter.plot()


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
