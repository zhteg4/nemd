import io
import pandas as pd
from scipy import constants

from nemd import lammpsin


class Log(lammpsin.In):

    LJ = 'LJ'
    METAL = 'metal'
    REAL = lammpsin.In.REAL
    DEFAULT_UNIT = lammpsin.In.REAL

    DEFAULT_TIMESTEP = {LJ: 0.005, REAL: 1., METAL: 0.001}
    TIME_UNITS = {lammpsin.In.REAL: constants.femto, METAL: constants.pico}
    TIME_TO_PS = {x: y / constants.pico for x, y in TIME_UNITS.items()}

    FS = 'fs'
    PS = 'ps'
    N_STEP = 'n'
    KELVIN = 'K'
    ATMOSPHERES = 'atmospheres'
    BARS = 'bars'
    KCAL_MOL = 'kcal/mol'
    EV = 'eV'
    ANGSTROMS = 'Angstroms'
    ANGSTROMS_CUBED = 'Angstroms^3'
    TIME = 'time'
    STEP = 'Step'
    TEMP = 'Temp'
    E_PAIR = 'E_pair'
    E_MOL = 'E_mol'
    TOTENG = 'TotEng'
    PRESS = 'Press'
    VOLUME = 'Volume'
    TIME_UNITS = {REAL: FS, METAL: PS}
    STEP_UNITS = {REAL: N_STEP, METAL: N_STEP}
    TEMP_UNITS = {REAL: KELVIN, METAL: KELVIN}
    ENG_UNITS = {REAL: KCAL_MOL, METAL: EV}
    PRESS_UNITS = {REAL: ATMOSPHERES, METAL: BARS}
    VOLUME_UNITS = {REAL: ANGSTROMS_CUBED, METAL: ANGSTROMS_CUBED}
    THERMO_UNITS = {
        TIME: TIME_UNITS,
        STEP: STEP_UNITS,
        TEMP: TEMP_UNITS,
        E_PAIR: ENG_UNITS,
        E_MOL: ENG_UNITS,
        TOTENG: ENG_UNITS,
        PRESS: PRESS_UNITS,
        VOLUME: VOLUME_UNITS
    }

    def __init__(self, filename):
        """
        :param filename: LAMMPS log file name
        :type filename: str
        """
        self.filename = filename
        self.unit = self.DEFAULT_UNIT
        self.timestep = None
        self.thermo = pd.DataFrame()

    def run(self):
        """
        Main method to parse the LAMMPS log file.
        """
        self.parse()
        self.finish()

    def parse(self):
        """
        Parse the LAMMPS log file to extract the thermodynamic data.
        """
        blk = []
        with open(self.filename) as fh:
            while line := fh.readline():
                if line.startswith('Loop time of'):
                    # Finishing up previous thermo block
                    data = pd.read_csv(io.StringIO(''.join(blk)), sep=r'\s+')
                    self.thermo = pd.concat((self.thermo, data))
                    blk = []
                elif blk:
                    # Inside thermo block: skip lines from fix rigid outputs
                    if not line.startswith(('SHAKE', 'Bond')):
                        blk.append(line)
                elif line.startswith('Per MPI rank memory allocation'):
                    # Start a new block
                    blk = [fh.readline()]
                # Other information outside the thermo block
                elif line.startswith(self.UNITS):
                    self.unit = line.strip(self.UNITS).strip()
                elif line.startswith(self.TIMESTEP):
                    self.timestep = line.strip(self.TIMESTEP).strip()

    def finish(self):
        """
        Finish update thermodynamic data by unit conversion, time conversion,
        column renaming, index setting, and etc.
        """
        timestep = self.DEFAULT_TIMESTEP[
            self.unit] if self.timestep is None else self.timestep
        time = self.thermo[self.STEP] * float(timestep)
        self.thermo.set_index(time)
        self.thermo.index.name = f"{self.TIME} ({self.TIME_UNITS[self.unit]})"
        self.thermo.drop(columns=self.STEP, inplace=True)
        self.thermo.columns = [
            f"{x} ({self.THERMO_UNITS[x][self.unit]})"
            for x in self.thermo.columns
        ]
