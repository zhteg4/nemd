import environutils
import math
import os
import numpy as np
import logutils
import units
from io import StringIO
import itertools
from dataclasses import dataclass
from matplotlib import pyplot as plt
from collections import namedtuple
from _collections import defaultdict

LogData = namedtuple('LogData', ['fix', 'data'])
FixCommand = namedtuple('FixCommand', ['id', 'group_id', 'style', 'args'])

AREA_LINE = 'The cross sectional area is %.6g Angstroms^2\n'
REX_AREA = 'The cross sectional area is (?P<name>\d*\.?\d*) Angstroms\^2\n'

NEMD_SRC = 'NEMD_SRC'
MODULE = 'module'
OPLSAA = 'oplsaa'
OPLSUA = 'oplsua'
MOLT_FF_EXT = '.lt'
RRM_EXT = '.prm'
FF = 'ff'

logger = logutils.createModuleLogger(file_path=__file__)


def log_debug(msg):

    if logger is None:
        return
    logger.debug(msg)


def get_ff(fn=None, name=OPLSAA, ext=MOLT_FF_EXT):
    if not fn:
        fn = f"{name}{ext}"
    return os.path.join(environutils.get_module_path(), FF, fn)


class chdir:

    def __init__(self, dirname):
        self.dirname = dirname
        self.original_dir = os.getcwd()

    def __enter__(self):
        if not os.path.exists(self.dirname):
            os.mkdir(self.dirname)
        os.chdir(self.dirname)

    def __exit__(self, exc_type, exc_value, exc_tb):
        os.chdir(self.original_dir)


@dataclass
class Processors:
    x: str
    y: str
    z: str

    def __post_init__(self):
        try_int = lambda x: int(x) if isinstance(x, str) and x.isdigit() else x
        self.x = try_int(self.x)
        self.y = try_int(self.y)
        self.z = try_int(self.z)


class LammpsInput(object):

    HASH = '#'
    # SUPPORTED COMMANDS
    PAIR_MODIFY = 'pair_modify'
    REGION = 'region'
    CHANGE_BOX = 'change_box'
    THERMO = 'thermo'
    GROUP = 'group'
    VELOCITY = 'velocity'
    DIHEDRAL_STYLE = 'dihedral_style'
    IMPROPER_STYLE = 'improper_style'
    COMPUTE = 'compute'
    THERMO_STYLE = 'thermo_style'
    READ_DATA = 'read_data'
    FIX = 'fix'
    DUMP_MODIFY = 'dump_modify'
    PAIR_STYLE = 'pair_style'
    PAIR_MODIFY = 'pair_modify'
    SPECIAL_BONDS = 'special_bonds'
    KSPACE_STYLE = 'kspace_style'
    KSPACE_MODIFY = 'kspace_modify'
    GEWALD = 'gewald'
    RUN = 'run'
    MINIMIZE = 'minimize'
    ANGLE_STYLE = 'angle_style'
    PROCESSORS = 'processors'
    VARIABLE = 'variable'
    BOND_STYLE = 'bond_style'
    NEIGHBOR = 'neighbor'
    DUMP = 'dump'
    NEIGH_MODIFY = 'neigh_modify'
    THERMO_MODIFY = 'thermo_modify'
    UNITS = 'units'
    ATOM_STYLE = 'atom_style'
    TIMESTEP = 'timestep'
    UNFIX = 'unfix'
    RESTART = 'restart'
    LOG = 'log'
    COMMANDS_KEYS = set([
        PAIR_MODIFY, REGION, CHANGE_BOX, THERMO, GROUP, VELOCITY,
        DIHEDRAL_STYLE, COMPUTE, THERMO_STYLE, READ_DATA, FIX, DUMP_MODIFY,
        PAIR_STYLE, RUN, MINIMIZE, ANGLE_STYLE, PROCESSORS, VARIABLE,
        BOND_STYLE, NEIGHBOR, DUMP, NEIGH_MODIFY, THERMO_MODIFY, UNITS,
        ATOM_STYLE, TIMESTEP, UNFIX, RESTART, LOG
    ])
    # Set parameters that need to be defined before atoms are created or read-in from a file.
    # The relevant commands are units, dimension, newton, processors, boundary, atom_style, atom_modify.
    # INITIALIZATION_KEYS = [
    #     UNITS, PROCESSORS, ATOM_STYLE, PAIR_STYLE, BOND_STYLE, ANGLE_STYLE,
    #     DIHEDRAL_STYLE
    # ]

    REAL = 'real'
    FULL = 'full'

    INITIALIZATION_ITEMS = {
        UNITS: set([REAL]),
        ATOM_STYLE: set([FULL]),
        PROCESSORS: Processors
    }

    # There are 3 ways to define the simulation cell and reserve space for force field info and fill it with atoms in LAMMPS
    # Read them in from (1) a data file or (2) a restart file via the read_data or read_restart commands
    # SYSTEM_DEFINITION_KEYS = [READ_DATA]
    SYSTEM_DEFINITION_ITEMS = {READ_DATA: str}

    # SIMULATION_SETTINGS_KEYS = [TIMESTEP, THERMO]
    TIMESTEP = 'timestep'
    THERMO = 'thermo'
    FIX = 'fix'
    AVE_CHUNK = 'ave/chunk'
    FILE = 'file'
    SIMULATION_SETTINGS_KEYS_ITEMS = {
        TIMESTEP: float,
        THERMO: int,
        FIX: FixCommand,
        LOG: str
    }

    ALL_ITEMS = {}
    ALL_ITEMS.update(INITIALIZATION_ITEMS)
    ALL_ITEMS.update(SYSTEM_DEFINITION_ITEMS)
    ALL_ITEMS.update(SIMULATION_SETTINGS_KEYS_ITEMS)

    def __init__(self, input_file):
        self.input_file = input_file
        self.lines = None
        self.commands = []
        self.cmd_items = {}
        self.is_debug = environutils.is_debug()

    def run(self):
        self.load()
        self.parser()

    def load(self):
        with open(self.input_file, 'r') as fh:
            self.raw_data = fh.read()

    def parser(self):
        self.loadCommands()
        self.setCmdKeys()
        self.setCmdItems()

    def loadCommands(self):
        commands = self.raw_data.split('\n')
        commands = [
            command.split() for command in commands
            if not command.startswith(self.HASH)
        ]
        self.commands = [command for command in commands if command]

    def setCmdKeys(self):
        self.cmd_keys = set([command[0] for command in self.commands])
        if not self.cmd_keys.issubset(self.COMMANDS_KEYS):
            unknown_keys = [
                key for key in self.data_keys if key not in self.COMMANDS_KEYS
            ]
            raise ValueError(f"{unknown_keys} are unknown.")

    def setCmdItems(self):
        for command in self.commands:
            cmd_key = command[0]
            cmd_values = command[1:]

            expected = self.ALL_ITEMS.get(cmd_key)
            if not expected:
                log_debug(f"{cmd_key} is not a known key.")
                continue
            if len(cmd_values) == 1:
                cmd_value = cmd_values[0]
                if isinstance(expected, set):
                    # e.g. units can be real, metal, lj, ... but not anything
                    if cmd_value not in expected:
                        raise ValueError(
                            f"{cmd_value} not in {expected} for {cmd_key}")
                    self.cmd_items[cmd_key] = cmd_value
                    continue

            if cmd_key == self.FIX:
                fix_command = expected(id=cmd_values[0],
                                       group_id=cmd_values[1],
                                       style=cmd_values[2],
                                       args=cmd_values[3:])
                self.cmd_items.setdefault(self.FIX.upper(),
                                          []).append(fix_command)
                continue

            if callable(expected):
                self.cmd_items[cmd_key] = expected(*cmd_values)

    def getTempFile(self):
        tempfile_basename = self.getTempFileBaseName()
        if tempfile_basename is None:
            return None
        return os.path.join(os.path.dirname(self.input_file),
                            tempfile_basename)

    def getEnergyFile(self):
        ene_file = self.cmd_items.get('log', None)
        if ene_file is None:
            return None
        return os.path.join(os.path.dirname(self.input_file), ene_file)

    def getTempFileBaseName(self):
        ave_chunk_comands = [
            x for x in self.cmd_items[self.FIX.upper()]
            if x.style == self.AVE_CHUNK
        ]
        if not ave_chunk_comands:
            return None
        ave_chunk_args = ave_chunk_comands[-1].args
        try:
            file_index = ave_chunk_args.index(self.FILE)
        except ValueError:
            return None
        try:
            return ave_chunk_args[file_index + 1]
        except IndexError:
            return None

    def getUnits(self):
        return self.cmd_items[self.UNITS]

    def getTimestep(self):
        return self.cmd_items[self.TIMESTEP]


class EnergyReader(object):

    THERMO = 'thermo'
    THERMO_SPACE = THERMO + ' '
    THERMO_STYLE = 'thermo_style'
    RUN = 'run'

    ENERGY_IN_KEY = 'Energy In (Kcal/mole)'
    ENERGY_OUT_KEY = 'Energy Out (Kcal/mole)'
    TIME_NS = 'Time (ns)'

    def __init__(self, energy_file, timestep):
        self.energy_file = energy_file
        self.timestep = timestep
        self.start_line_num = 1
        self.thermo_intvl = 1
        self.total_step_num = 1
        self.total_line_num = 1
        self.data_formats = ('float', 'float', 'float', 'float')
        self.data_type = None

    def run(self):
        self.setStartEnd()
        self.loadData()
        self.setUnits()
        self.setHeatflux()

    def write(self, filename):
        time = self.data['Time (ns)']
        ene_in = np.abs(self.data['Energy In (Kcal/mole)'])
        ene_out = self.data['Energy Out (Kcal/mole)']
        ene_data = np.concatenate((ene_in.reshape(1,
                                                  -1), ene_out.reshape(1, -1)))
        ene_data = np.transpose(ene_data)
        data = np.concatenate(
            (time.reshape(1, -1), ene_data.mean(axis=1).reshape(1, -1),
             ene_data.std(axis=1).reshape(1, -1)))
        data = np.transpose(data)
        col_titles = [
            'Time (ns)', 'Energy (Kcal/mole)',
            'Energy Standard Deviation (Kcal/mole)'
        ]
        np.savez(filename, data=data, header=','.join(col_titles))

    def setStartEnd(self):
        with open(self.energy_file, 'r') as file_energy:
            one_line = file_energy.readline()
            while not one_line.startswith('Step'):
                self.start_line_num += 1
                if one_line.startswith(self.THERMO_SPACE):
                    # thermo 1000
                    log_debug(one_line)
                    self.thermo_intvl = int(one_line.split()[-1])
                elif one_line.startswith(self.RUN):
                    log_debug(one_line)
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
        log_debug(
            f'Loading {self.total_line_num} lines of {self.energy_file} starting from line {self.start_line_num}'
        )
        try:
            self.data = np.loadtxt(self.energy_file,
                                   dtype=self.data_type,
                                   skiprows=self.start_line_num,
                                   max_rows=self.total_line_num)
        except ValueError as err:
            # Wrong number of columns at line 400003
            err_str = str(err)
            log_debug(err_str + f' in loading {self.energy_file}: {err_str}')
            self.total_line_num = int(
                err_str.split()[-1]) - self.start_line_num - 1
        else:
            return

        self.data = np.loadtxt(self.energy_file,
                               dtype=self.data_type,
                               skiprows=self.start_line_num,
                               max_rows=self.total_line_num)

    def setUnits(self):
        self.setTimeUnit()
        self.setTempUnit()
        self.setEnergyUnit()

    def setTimeUnit(self, unit='ns', reset=True):
        orig_time_key = self.data.dtype.names[0]
        if reset:
            self.data[orig_time_key] = self.data[orig_time_key] - self.data[
                orig_time_key][0]
        self.data[orig_time_key] = self.data[orig_time_key] * self.timestep
        time_key = 'Time'
        if unit == 'ns':
            self.data[
                orig_time_key] = self.data[orig_time_key] / units.NANO2FETO
            time_key += ' (ns)'
        self.data.dtype.names = tuple([time_key] +
                                      list(self.data.dtype.names[1:]))

    def setTempUnit(self, unit='K'):
        temp_key = 'Temperature (K)'
        self.data.dtype.names = tuple([self.data.dtype.names[0]] + [temp_key] +
                                      list(self.data.dtype.names[2:]))

    def setEnergyUnit(self):

        self.data.dtype.names = tuple(
            list(self.data.dtype.names[:2]) +
            [self.ENERGY_IN_KEY, self.ENERGY_OUT_KEY])

    def setHeatflux(self, qstart=0.2):
        start_idx = int(self.data.shape[0] * qstart)
        qdata = np.concatenate(
            (self.data[self.ENERGY_IN_KEY][..., np.newaxis],
             self.data[self.ENERGY_OUT_KEY][..., np.newaxis]),
            axis=1)
        sel_qdata = qdata[start_idx:, :]
        sel_q_mean = np.abs(sel_qdata).mean(axis=1)
        sel_time = self.data[self.TIME_NS][start_idx:]
        # Energy In (Kcal/mole) / Time (ns)
        self.slope, self.intercept = np.polyfit(sel_time, sel_q_mean, 1)
        fitted_q = np.polyval([self.slope, self.intercept], sel_time)
        self.fitted_data = np.concatenate(
            (sel_time[..., np.newaxis], fitted_q[..., np.newaxis]), axis=1)


def get_line_num(filename):

    def blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    with open(filename, "r", encoding="utf-8", errors='ignore') as f:
        line_num = sum(bl.count("\n") for bl in blocks(f))

    return line_num


class LammpsLogReader(object):

    FIX = 'fix'  # fix NPT all npt temp 0.1 0.1 25  x 0 0 2500  y 0 0 2500    z 0 0 2500
    STEP = 'Step'
    LOOP = 'Loop'
    LX = 'Lx'
    LY = 'Ly'
    LZ = 'Lz'

    def __init__(self, lammps_log, cross_sectional_area=None):
        self.lammps_log = lammps_log
        self.cross_sectional_area = cross_sectional_area
        self.all_data = []

    def run(self):
        self.loadAllData()
        self.setCrossSectionalArea()
        self.plot()

    def loadAllData(self):
        with open(self.lammps_log, "r", encoding="utf-8",
                  errors='ignore') as file_log:
            fix_line = None
            line = file_log.readline()
            while line:
                line = file_log.readline()
                if line.startswith(self.FIX):
                    fix_line = line
                if not line.startswith(self.STEP):
                    continue

                names = line.split()
                formats = [int if x == self.STEP else float for x in names]
                data_type = {'names': names, 'formats': formats}
                data_type[self.STEP] = int

                data_str = ""
                line = file_log.readline()
                while line and not line.startswith(self.LOOP):
                    data_str += line
                    line = file_log.readline()
                data = np.loadtxt(StringIO(data_str), dtype=data_type)
                self.all_data.append(LogData(fix=fix_line, data=data))

    def setCrossSectionalArea(self,
                              first_dimension_lb=LY,
                              second_dimension_lb=LZ):
        if self.cross_sectional_area is not None:
            return

        d1_length, d2_length = None, None
        for data in reversed(self.all_data):
            try:
                d1_length = data.data[first_dimension_lb]
            except ValueError:
                continue

            try:
                d2_length = data.data[second_dimension_lb]
            except ValueError:
                d1_length = None
                continue

            if d1_length is not None and d2_length is not None:
                break

        if any([d1_length is None, d2_length is None]):
            raise ValueError(
                "Please define a cross-sectional area via -cross_sectional_area"
            )
        self.cross_sectional_area = np.mean(d1_length * d2_length)

    def plot(self):

        if not environutils.is_interactive():
            return

        names = set([y for x in self.all_data for y in x.data.dtype.names])
        names.remove(self.STEP)
        fig_ncols = 2
        fig_nrows = math.ceil(len(names) / fig_ncols)
        self.fig = plt.figure(figsize=(12, 7))
        self.axises = []
        data = self.all_data[-1]
        for fig_index, name in enumerate(names, start=1):
            axis = self.fig.add_subplot(fig_nrows, fig_ncols, fig_index)
            self.axises.append(axis)
            for data in self.all_data:
                try:
                    y_data = data.data[name]
                except ValueError:
                    continue
                try:
                    line, = axis.plot(data.data[self.STEP], y_data)
                except:
                    import pdb
                    pdb.set_trace()
                axis.set_ylabel(name)

        self.fig.legend(axis.lines,
                        [x.fix.replace('\t', '') for x in self.all_data],
                        loc="upper right",
                        ncol=3,
                        prop={'size': 8.3})
        self.fig.tight_layout(
            rect=(0.0, 0.0, 1.0, 1.0 -
                  self.fig.legends[0].handleheight / self.fig.get_figheight()))

        input('Showing the lammps log plot. Press any keys to continue...')


class TempReader(object):

    def __init__(self, temp_file, block_num=5):
        self.temp_file = temp_file
        self.block_num = block_num
        self.data = None
        self.frame_num = None
        self.fitted_data = None
        self.slope = None
        self.intercept = None

    def run(self):
        self.load()
        self.setTempGradient()

    def write(self, filename):
        coords = self.data[:, 1, -1]
        temps = self.data[:, 3, -1]
        block_temps = self.data[:, 3, :-1]
        std_temps = np.std(block_temps, axis=1)
        data = np.concatenate((coords.reshape(1, -1), temps.reshape(1, -1),
                               std_temps.reshape(1, -1)))
        data = np.transpose(data)
        col_titles = [
            'Coordinates (Angstrom)', 'Temperature (K)',
            'Temperature Standard Deviation (K)'
        ]
        np.savez(filename, data=data, header=','.join(col_titles))

    def load(self):

        line_num = get_line_num(self.temp_file)
        header_line_num = 3
        with open(self.temp_file, 'r') as file_temp:
            step_nbin_nave = np.loadtxt(file_temp,
                                        skiprows=header_line_num,
                                        max_rows=1)
            nbin = int(step_nbin_nave[1])
            self.frame_num = math.floor(
                (line_num - header_line_num) / (nbin + 1))
            frame_per_block = math.floor(self.frame_num / self.block_num)
            self.data = np.zeros((nbin, 4, self.block_num + 1))
            for data_index in range(self.block_num):
                for iframe in range(frame_per_block):
                    tmp_data = np.array(np.loadtxt(file_temp, max_rows=nbin))
                    self.data[:, :, data_index] += (tmp_data / frame_per_block)
                    file_temp.readline()
            self.data[:, :, -1] = self.data[:, :, :self.block_num].mean(axis=2)

    def setTempGradient(self, crange=(0.15, 0.85)):
        coords = self.data[:, 1, -1]
        temps = self.data[:, 3, -1]
        coord_num = len(coords)
        indexes = [int(coord_num * x) for x in crange]
        sel_coords = coords[indexes[0]:indexes[-1] + 1]
        sel_temps = temps[indexes[0]:indexes[-1] + 1]
        # Temperature (K) / Coordinate (Angstrom)
        self.slope, self.intercept = np.polyfit(sel_coords, sel_temps, 1)
        fitted_temps = np.polyval([self.slope, self.intercept], sel_coords)
        self.fitted_data = np.concatenate(
            (sel_coords[..., np.newaxis], fitted_temps[..., np.newaxis]),
            axis=1)
