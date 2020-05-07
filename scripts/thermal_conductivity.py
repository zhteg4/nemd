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


class EnergyFileReader(object):
    def __init__(self, energy_file):
        self.energy_file = energy_file
        self.start_line_num = 1
        self.thermo_intvl = 1
        self.total_step_num = 1
        self.total_line_num = 1
        self.data_type = None

    def run(self):
        self.setStartEnd()
        self.loadData()

    def setStartEnd(self):
        with open(self.energy_file, 'r') as file_energy:
            one_line = file_energy.readline()
            while not one_line.startswith('Step'):
                self.start_line_num += 1
                if one_line.startswith(THERMO_SPACE):
                    # thermo 1000
                    log_debug(one_line)
                    self.thermo_intvl = int(one_line.split()[-1])
                elif one_line.startswith(RUN):
                    log_debug(one_line)
                    # run 400000000
                    self.total_step_num = int(one_line.split()[-1])
                one_line = file_energy.readline()
            self.total_line_num = math.floor(self.total_step_num /
                                             self.thermo_intvl)
            data_names = one_line.split()
            data_formats = ('int', 'float', 'float', 'float')
            self.data_type = {'names': data_names, 'formats': data_formats}

    def loadData(self):
        #log_debug(f'Loading {self.total_line_num} lines of {self.energy_file} starting from line {self.start_line_num}')
        try:
            self.data = np.loadtxt(self.energy_file,
                                   dtype=self.data_type,
                                   skiprows=self.start_line_num,
                                   max_rows=self.total_line_num)
        except ValueError as err:
            # Wrong number of columns at line 400003
            err_str = str(err)
            log_debug(err_str + f' in loading {self.energy_file}')
            self.total_line_num = int(
                err_str.split()[-1]) - self.start_line_num - 1
        else:
            return

        self.data = np.loadtxt(self.energy_file,
                               dtype=self.data_type,
                               skiprows=self.start_line_num,
                               max_rows=self.total_line_num)


def load_temp_file(temp_file):
    with open(temp_file, 'r') as file_temp:
        step_nbin_nave = np.loadtxt(file_temp, skiprows=3, max_rows=1)
        nbin = int(step_nbin_nave[1])
        data = np.zeros((nbin, 4))
        data_num = 0
        while file_temp:
            data_num += 1
            data += np.array(np.loadtxt(file_temp, max_rows=nbin))
            if not file_temp.readline():
                return data / data_num


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
