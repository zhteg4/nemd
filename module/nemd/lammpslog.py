import io
import pandas as pd
from scipy import constants

from nemd import symbols
from nemd import lammpsin


class Log(lammpsin.In):
    """
    Class to parse LAMMPS log file and extract data.
    """

    LJ = lammpsin.In.LJ
    METAL = lammpsin.In.METAL
    REAL = lammpsin.In.REAL
    DEFAULT_UNIT = REAL

    DEFAULT_TIMESTEP = {LJ: 0.005, REAL: 1., METAL: 0.001}
    TIME_UNITS = {lammpsin.In.REAL: constants.femto, METAL: constants.pico}
    TIME_TO_PS = {x: y / constants.pico for x, y in TIME_UNITS.items()}

    FS = symbols.FS
    PS = symbols.PS
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
        self.read()
        self.finalize()

    def read(self):
        """
        Read the LAMMPS log file to extract the thermodynamic data.
        """
        with open(self.filename) as fh:
            blk = []
            while line := fh.readline():
                if line.startswith('Loop time of'):
                    # Finishing up previous thermo block
                    data = pd.read_csv(io.StringIO(''.join(blk)), sep=r'\s+')
                    self.thermo = pd.concat((self.thermo, data))
                    blk = []
                elif blk:
                    # Inside thermo block: skip lines from fix rigid outputs
                    if not line.startswith(('SHAKE', 'Bond', 'Angle')):
                        blk.append(line)
                elif line.startswith('Per MPI rank memory allocation'):
                    # Start a new block
                    blk = [fh.readline()]
                # Other information outside the thermo block
                elif line.startswith(self.UNITS):
                    self.unit = line.strip(self.UNITS).strip()
                elif line.startswith(self.TIMESTEP):
                    self.timestep = int(line.strip(self.TIMESTEP).strip())
        if blk:
            # Finishing up the last running thermo block
            data = pd.read_csv(io.StringIO(''.join(blk)), sep=r'\s+')
            self.thermo = pd.concat((self.thermo, data))
        if self.timestep is None:
            self.timestep = self.DEFAULT_TIMESTEP[self.unit]

    def finalize(self):
        """
        Finalize the extracted data by unit conversion, time conversion,
        column renaming, index setting, and etc.
        """
        self.thermo[self.STEP] = self.thermo[self.STEP] * float(self.timestep)
        self.thermo.set_index(self.STEP, inplace=True)
        time_unit = self.TIME_UNITS[self.unit]
        if time_unit != self.PS:
            self.thermo.index *= self.TIME_TO_PS[self.unit]
        self.thermo.index.name = symbols.TIME_LB.format(unit=self.PS)
        self.thermo.columns = [
            f"{x} ({self.THERMO_UNITS[x][self.unit]})"
            for x in self.thermo.columns
        ]
