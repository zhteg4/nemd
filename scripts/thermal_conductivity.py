import sys
import logutils
import os
import parserutils
import fileutils

FLAG_IN_FILE = 'in_file'
FLAG_TEMP_FILE = 'temp_file'
FlAG_ENEGER_FILE = 'energy_file'

NANO2FETO = 1E6

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
    units = in_reader.cmd_items[in_reader.UNITS]
    log(f"{units} is the units.")
    timestep = in_reader.cmd_items[in_reader.TIMESTEP]
    log(f"{timestep} is the timestep.")
    energy_reader = fileutils.EnergyReader(options.energy_file)
    energy_reader.run()
    log(f"Found {energy_reader.total_step_num} steps of energy logging, "
        f"corresponding to {energy_reader.total_step_num * timestep / NANO2FETO} ns"
        )
    block_num = 5
    temp_dat, temp_dat_num = fileutils.load_temp(options.temp_file,
                                                 block_num=block_num)
    log(f"Every {int(temp_dat_num / block_num)} successive temperature profiles out of "
        f"{temp_dat_num} are block averaged")

    from matplotlib import pyplot as plt
    fig = plt.figure()
    nrows, ncols = 2, 1
    axis = fig.add_subplot(nrows, ncols, 1)
    axis.plot(energy_reader.data['Step'] * timestep / NANO2FETO,
              -energy_reader.data['f_7'],
              label='In')
    axis.plot(energy_reader.data['Step'] * timestep / NANO2FETO,
              energy_reader.data['f_8'],
              label='Out')
    axis.set_xlabel('Time (ns)')
    axis.set_ylabel('Energy ()')

    axis = fig.add_subplot(nrows, ncols, 2)
    nrow, ncol, nblock = temp_dat.shape
    for iblock in range(nblock - 1):
        axis.plot(temp_dat[:, 1, iblock],
                  temp_dat[:, 3, iblock],
                  '.',
                  label=f'Block {iblock}')
    axis.plot(temp_dat[:, 1, -1], temp_dat[:, 3, -1], label='Average')
    axis.legend(loc='upper right', prop={'size': 6})
    axis.set_ylim([270, 330])
    axis.set_xlabel('Coordinate (Angstrom)')
    axis.set_ylabel('Temperature (K)')
    plt.tight_layout()
    fig.show()

    input()


if __name__ == "__main__":
    main(sys.argv[1:])
