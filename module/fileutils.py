import math
import numpy as np


class EnergyReader(object):

    THERMO = 'thermo'
    THERMO_SPACE = THERMO + ' '
    THERMO_STYLE = 'thermo_style'
    RUN = 'run'

    def __init__(self, energy_file):
        self.energy_file = energy_file
        self.start_line_num = 1
        self.thermo_intvl = 1
        self.total_step_num = 1
        self.total_line_num = 1
        self.data_formats = ('int', 'float', 'float', 'float')
        self.data_type = None

    def run(self):
        self.setStartEnd()
        self.loadData()

    def setStartEnd(self):
        with open(self.energy_file, 'r') as file_energy:
            one_line = file_energy.readline()
            while not one_line.startswith('Step'):
                self.start_line_num += 1
                if one_line.startswith(self.THERMO_SPACE):
                    # thermo 1000
                    # log_debug(one_line)
                    self.thermo_intvl = int(one_line.split()[-1])
                elif one_line.startswith(self.RUN):
                    # log_debug(one_line)
                    # run 400000000
                    self.total_step_num = int(one_line.split()[-1])
                one_line = file_energy.readline()
            self.total_line_num = math.floor(self.total_step_num /
                                             self.thermo_intvl)
            data_names = one_line.split()
            self.data_type = {
                'names': data_names,
                'formats': self.data_formats
            }

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
            #log_debug(err_str + f' in loading {self.energy_file}')
            self.total_line_num = int(
                err_str.split()[-1]) - self.start_line_num - 1
        else:
            return

        self.data = np.loadtxt(self.energy_file,
                               dtype=self.data_type,
                               skiprows=self.start_line_num,
                               max_rows=self.total_line_num)


def load_temp(temp_file):
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
