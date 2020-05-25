import environutils
import math
from dataclasses import dataclass
import numpy as np
import logutils
import units

logger = logutils.createModuleLogger()


def log_debug(msg):

    if logger is None:
        return
    logger.debug(msg)


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
    COMPUTE = 'compute'
    THERMO_STYLE = 'thermo_style'
    READ_DATA = 'read_data'
    FIX = 'fix'
    DUMP_MODIFY = 'dump_modify'
    PAIR_STYLE = 'pair_style'
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
    SIMULATION_SETTINGS_KEYS_ITEMS = {
        TIMESTEP: float,
        THERMO: int,
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
                    if cmd_value not in expected:
                        raise ValueError(
                            f"{cmd_value} not in {expected} for {cmd_key}")
                    self.cmd_items[cmd_key] = cmd_value
                    continue
                if callable(expected):
                    self.cmd_items[cmd_key] = expected(cmd_value)
            self.cmd_items[cmd_key] = expected(*cmd_values)


class EnergyReader(object):

    THERMO = 'thermo'
    THERMO_SPACE = THERMO + ' '
    THERMO_STYLE = 'thermo_style'
    RUN = 'run'

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
            log_debug(err_str + f' in loading {self.energy_file}')
            self.total_line_num = int(
                err_str.split()[-1]) - self.start_line_num - 1
        else:
            return

        self.data = np.loadtxt(self.energy_file,
                               dtype=self.data_type,
                               skiprows=self.start_line_num,
                               max_rows=self.total_line_num)

    def setTimeUnit(self, unit='ns'):
        orig_time_key = self.data.dtype.names[0]
        self.data[orig_time_key] = self.data[orig_time_key] * self.timestep
        time_key = 'Time'
        if unit == 'ns':
            self.data[
                orig_time_key] = self.data[orig_time_key] / units.NANO2FETO
            time_key += ' ns'
        self.data.dtype.names = tuple([time_key] +
                                      list(self.data.dtype.names[1:]))

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
        energy_in_key = 'Energy In (Kcal/mole)'
        energy_out_key = 'Energy Out (Kcal/mole)'
        self.data.dtype.names = tuple(
            list(self.data.dtype.names[:2]) + [energy_in_key, energy_out_key])


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b


def load_temp(temp_file, block_num=5):

    with open(temp_file, "r", encoding="utf-8", errors='ignore') as f:
        line_num = sum(bl.count("\n") for bl in blocks(f))

    header_line_num = 3
    with open(temp_file, 'r') as file_temp:
        step_nbin_nave = np.loadtxt(file_temp,
                                    skiprows=header_line_num,
                                    max_rows=1)
        nbin = int(step_nbin_nave[1])
        frame_num = math.floor((line_num - header_line_num) / (nbin + 1))
        frame_per_block = math.floor(frame_num / block_num)
        data = np.zeros((nbin, 4, block_num + 1))
        for data_index in range(block_num):
            for iframe in range(frame_per_block):
                tmp_data = np.array(np.loadtxt(file_temp, max_rows=nbin))
                data[:, :, data_index] += (tmp_data / frame_per_block)
                file_temp.readline()
        data[:, :, -1] = data[:, :, 1:block_num].mean(axis=2)
        return data, frame_num
