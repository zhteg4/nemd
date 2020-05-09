import sys
import math
import os
import parserutils
import numpy as np
import logging

FLAG_TEMP_FILE = 'temp_file'
FlAG_ENEGER_FILE = 'energy_file'

THERMO = 'thermo'
THERMO_SPACE = THERMO + ' '
THERMO_STYLE = 'thermo_style'
RUN = 'run'

JOBNAME = os.path.basename(__file__).split('.')[0]


def log_debug(msg):
    if logger:
        logger.debug(msg)


def createLogger(jobname, verbose=True):
    logger = logging.getLogger(jobname)
    hdlr = logging.FileHandler(jobname + '-driver.log')
    if verbose:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.setLevel(logging.DEBUG)
    logger.addHandler(hdlr)
    return logger





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
    jobname = JOBNAME
    logger = createLogger(jobname=jobname)
    options = validate_options(argv)
    energy_reader = EnergyFileReader(options.energy_file)
    energy_reader.run()
    temp_dat = load_temp_file(options.temp_file)


if __name__ == "__main__":
    main(sys.argv[1:])
