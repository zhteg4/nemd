import sys
import logutils
import os
import parserutils
import fileutils

FLAG_TEMP_FILE = 'temp_file'
FlAG_ENEGER_FILE = 'energy_file'

JOBNAME = os.path.basename(__file__).split('.')[0]


def log_debug(msg):
    if logger:
        logger.debug(msg)


def get_parser():
    parser = parserutils.get_parser(
        description=
        'Calculate thermal conductivity using non-equilibrium molecular dynamics.'
    )
    parser.add_argument(FLAG_TEMP_FILE,
                        metavar=FLAG_TEMP_FILE.upper(),
                        help='')
    parser.add_argument(FlAG_ENEGER_FILE,
                        metavar=FlAG_ENEGER_FILE.upper(),
                        type=parserutils.type_file,
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
    energy_reader = fileutils.EnergyReader(options.energy_file)
    energy_reader.run()
    temp_dat = fileutils.load_temp(options.temp_file)


if __name__ == "__main__":
    main(sys.argv[1:])
