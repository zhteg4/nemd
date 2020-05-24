import sys
import logutils
import os
import units
import parserutils
import fileutils
import plotutils

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
    return parser


def validate_options(argv):
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


logger = None


def main(argv):
    global logger

    logger = logutils.createDriverLogger(jobname=JOBNAME)
    options = validate_options(argv)
    logutils.logOptions(logger, options)

    in_reader = fileutils.LammpsInput(options.in_file)
    in_reader.run()
    lammps_units = in_reader.cmd_items[in_reader.UNITS]
    log(f"{lammps_units} is the units.")
    timestep = in_reader.cmd_items[in_reader.TIMESTEP]
    log(f"{timestep} is the timestep.")
    energy_reader = fileutils.EnergyReader(options.energy_file, timestep)
    energy_reader.run()
    log(f"Found {energy_reader.total_step_num} steps of energy logging, "
        f"corresponding to {energy_reader.total_step_num * timestep / units.NANO2FETO} ns"
        )
    block_num = 5
    temp_data, temp_dat_num = fileutils.load_temp(options.temp_file,
                                                 block_num=block_num)
    log(f"Every {int(temp_dat_num / block_num)} successive temperature profiles out of "
        f"{temp_dat_num} are block averaged")
    temp_ene_plotter = plotutils.TempEnePlotter(temp_data, energy_reader.data)
    temp_ene_plotter.plot()





if __name__ == "__main__":
    main(sys.argv[1:])
