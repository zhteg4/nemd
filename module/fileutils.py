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

TYPE_ID = 'type_id'
ATOM_ID = 'atom_id'

logger = logutils.createModuleLogger()

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


def get_src():
    return os.environ.get(NEMD_SRC)


def get_module():
    return os.path.join(get_src(), MODULE)


def get_ff(fn=None, name=OPLSAA, ext=MOLT_FF_EXT):
    if not fn:
        fn = f"{name}{ext}"
    return os.path.join(get_module(), FF, fn)


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


class LammpsWriter(LammpsInput):
    IN_EXT = '.in'
    DATA_EXT = '.data'
    LAMMPS_DESCRIPTION = 'LAMMPS Description'

    ATOMS = 'atoms'
    BONDS = 'bonds'
    ANGLES = 'angles'
    DIHEDRALS = 'dihedrals'
    IMPROPERS = 'impropers'

    ATOM_TYPES = 'atom types'
    BOND_TYPES = 'bond types'
    ANGLE_TYPES = 'angle types'
    DIHEDRAL_TYPES = 'dihedral types'
    IMPROPER_TYPES = 'improper types'

    XLO_XHI = 'xlo xhi'
    YLO_YHI = 'ylo yhi'
    ZLO_ZHI = 'zlo zhi'
    LO_HI = [XLO_XHI, YLO_YHI, ZLO_ZHI]

    MASSES = 'Masses'
    PAIR_COEFFS = 'Pair Coeffs'
    BOND_COEFFS = 'Bond Coeffs'
    ANGLE_COEFFS = 'Angle Coeffs'
    DIHEDRAL_COEFFS = 'Dihedral Coeffs'

    LJ_CUT_COUL_LONG = 'lj/cut/coul/long'
    LJ_CUT = 'lj/cut'

    def __init__(self, ff, jobname, mols=None, lj_cut=11., coul_cut=11.):
        self.ff = ff
        self.jobname = jobname
        self.mols = mols
        self.lj_cut = lj_cut
        self.coul_cut = coul_cut
        self.lammps_in = self.jobname + self.IN_EXT
        self.lammps_data = self.jobname + self.DATA_EXT
        self.units = 'real'
        self.atom_style = 'full'
        self.bond_style = 'harmonic'
        self.angle_style = 'harmonic'
        self.dihedral_style = 'opls'
        self.improper_style = 'harmonic'
        self.pair_style = {
            self.LJ_CUT_COUL_LONG:
            f"{self.LJ_CUT_COUL_LONG} {self.lj_cut} {self.coul_cut}",
            self.LJ_CUT: f"{self.LJ_CUT} {self.lj_cut}"
        }
        self.pair_modify = {'mix': 'geometric'}
        self.special_bonds = {
            'lj/coul': (
                0.0,
                0.0,
                0.5,
            )
        }
        self.kspace_style = {'pppm': 0.0001}

    def writeLammpsIn(self):
        with open(self.lammps_in, 'w') as fh:
            fh.write(f"{self.UNITS} {self.units}\n")
            fh.write(f"{self.ATOM_STYLE} {self.atom_style}\n")
            fh.write(f"{self.BOND_STYLE} {self.bond_style}\n")
            fh.write(f"{self.ANGLE_STYLE} {self.angle_style}\n")
            fh.write(f"{self.DIHEDRAL_STYLE} {self.dihedral_style}\n")
            fh.write(f"{self.IMPROPER_STYLE} {self.improper_style}\n")
            pair_style = self.LJ_CUT_COUL_LONG if self.hasCharge(
            ) else self.LJ_CUT
            fh.write(f"{self.PAIR_STYLE} {self.pair_style[pair_style]}\n")
            fh.write(
                f"{self.PAIR_MODIFY} {' '.join([(x,y) for x, y in self.pair_modify.items()][0])}\n"
            )
            special_bond = [
                f"{x} {' '.join(map(str, y))}"
                for x, y in self.special_bonds.items()
            ][0]
            fh.write(f"{self.SPECIAL_BONDS} {special_bond}\n")
            if self.hasCharge():
                kspace_style = [
                    f"{x} {y}" for x, y in self.kspace_style.items()
                ][0]
                fh.write(f"{self.KSPACE_STYLE} {kspace_style}\n")
            fh.write(f"{self.READ_DATA} {self.lammps_data}\n")

            fh.write("minimize 1.0e-4 1.0e-6 100 1000")

    def hasCharge(self, default=True):
        if self.mols is None:
            return default
        charges = [
            self.ff.charges[y.GetIntProp(TYPE_ID)] for x in self.mols.values()
            for y in x.GetAtoms()
        ]
        return any(charges)

    def writeLammpsData(self):

        with open(self.lammps_data, 'w') as self.data_fh:
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writePairCoeffs()
            self.writeBondCoeffs()
            self.writeAngleCoeffs()
            self.writeDihedralCoeffs()
            self.writeAtoms()
            self.writeBonds()
            self.writeAngles()
            self.writeDihedrals()

    def writeDescription(self):
        if self.mols is None:
            raise ValueError(f"Mols are not set.")

        self.data_fh.write(f"{self.LAMMPS_DESCRIPTION}\n\n")
        atoms = [len(x.GetAtoms()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(atoms)} {self.ATOMS}\n")
        bonds = [len(x.GetBonds()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(bonds)} {self.BONDS}\n")
        neighbors = [
            len(y.GetNeighbors()) for x in self.mols.values()
            for y in x.GetAtoms()
        ]
        # FIXME: I guess improper angles may reduce this num
        angles = [max(0, x - 1) for x in neighbors]
        self.data_fh.write(f"{sum(angles)} {self.ANGLES}\n")
        # FIXME: dihedral and improper are set to be zeros at this point
        self.data_fh.write(f"0 {self.DIHEDRALS}\n")
        self.data_fh.write(f"0 {self.IMPROPERS}\n\n")

    def writeTopoType(self):
        self.data_fh.write(f"{len(self.ff.atoms)} {self.ATOM_TYPES}\n")
        self.data_fh.write(f"{len(self.ff.bonds)} {self.BOND_TYPES}\n")
        self.data_fh.write(f"{len(self.ff.angles)} {self.ANGLE_TYPES}\n")
        self.data_fh.write(f"{len(self.ff.dihedrals)} {self.DIHEDRAL_TYPES}\n")
        self.data_fh.write(
            f"{len(self.ff.impropers)} {self.IMPROPER_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        if min_box is None:
            min_box = (20., 20., 20.,) # yapf: disable
        if buffer is None:
            buffer = (2., 2., 2.,) # yapf: disable
        xyzs = np.concatenate(
            [x.GetConformer(0).GetPositions() for x in self.mols.values()])
        box = xyzs.max(axis=0) - xyzs.min(axis=0) + buffer
        box_hf = [max([x, y]) / 2. for x, y in zip(box, min_box)]
        centroid = xyzs.mean(axis=0)
        for dim in range(3):
            self.data_fh.write(
                f"{centroid[dim]-box_hf[dim]:.2f} {centroid[dim]+box_hf[dim]:.2f} {self.LO_HI[dim]}\n"
            )
        self.data_fh.write("\n")

    def writeMasses(self):
        self.data_fh.write(f"{self.MASSES}\n\n")
        for atom_id, atom in self.ff.atoms.items():
            self.data_fh.write(f"{atom_id} {atom.mass} # {atom.description}\n")
        self.data_fh.write(f"\n")

    def writePairCoeffs(self):
        self.data_fh.write(f"{self.PAIR_COEFFS}\n\n")
        for atom in self.ff.atoms.values():
            vdw = self.ff.vdws[atom.id]
            self.data_fh.write(f"{atom.id} {atom.id} {vdw.ene} {vdw.dist}\n")
        self.data_fh.write("\n")

    def writeBondCoeffs(self):
        self.data_fh.write(f"{self.BOND_COEFFS}\n\n")
        for bond in self.ff.bonds.values():
            self.data_fh.write(f"{bond.id}  {bond.ene} {bond.dist}\n")
        self.data_fh.write("\n")

    def writeAngleCoeffs(self):
        self.data_fh.write(f"{self.ANGLE_COEFFS}\n\n")
        for angle in self.ff.angles.values():
            self.data_fh.write(f"{angle.id} {angle.ene} {angle.angle}\n")
        self.data_fh.write("\n")

    def writeDihedralCoeffs(self):
        self.data_fh.write(f"{self.DIHEDRAL_COEFFS}\n\n")
        for dihedral in self.ff.dihedrals.values():
            params = [0., 0., 0., 0.]
            for ene_ang_n in dihedral.constants:
                params[ene_ang_n.n_parm - 1] = ene_ang_n.ene * 2
                if (ene_ang_n.angle == 180.) ^ (ene_ang_n.n_parm in (
                        2,
                        4,
                )):
                    params[ene_ang_n.n_parm] *= -1
            self.data_fh.write(
                f"{dihedral.id}  {' '.join(map(str, params))}\n")
        self.data_fh.write("\n")

    def writeAtoms(self):
        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        atom_id = 0
        for mol_id, mol in self.mols.items():
            conformer = mol.GetConformer()
            for atom in mol.GetAtoms():
                atom_id += 1
                atom.SetIntProp(ATOM_ID, atom_id)
                type_id = atom.GetIntProp(TYPE_ID)
                xyz = conformer.GetAtomPosition(atom.GetIdx())
                xyz = ' '.join(map(lambda x: f'{x:.3f}', xyz))
                charge = self.ff.charges[type_id]
                self.data_fh.write(
                    f"{atom_id} {mol_id} {type_id} {charge} {xyz}\n")
        self.data_fh.write(f"\n")

    def writeBonds(self):
        self.data_fh.write(f"{self.BONDS.capitalize()}\n\n")
        bond_id = 0
        for mol in self.mols.values():
            for bond in mol.GetBonds():
                bond_id += 1
                bonded_atoms = [bond.GetBeginAtom(), bond.GetEndAtom()]
                bonded_atoms = sorted(bonded_atoms,
                                      key=lambda x: x.GetIntProp(TYPE_ID))
                atoms_types = [x.GetIntProp(TYPE_ID) for x in bonded_atoms]
                matches = [
                    x for x in self.ff.bonds.values()
                    if [x.id1, x.id2] == atoms_types
                ]
                if not matches:
                    raise ValueError(
                        f"Cannot find params for bond between atom {b_type_id} and {s_type_id}."
                    )
                bond = matches[0]
                self.data_fh.write(
                    f"{bond_id} {bond.id} {bonded_atoms[0].GetIntProp(ATOM_ID)} {bonded_atoms[1].GetIntProp(ATOM_ID)}\n"
                )
        self.data_fh.write(f"\n")

    def writeAngles(self):
        self.data_fh.write(f"{self.ANGLES.capitalize()}\n\n")
        angle_id = 0
        for mol in self.mols.values():
            for atom in mol.GetAtoms():
                for atoms in self.getAngleAtoms(atom):
                    angle_id += 1
                    type_ids = [x.GetIntProp(TYPE_ID) for x in atoms]
                    matches = [
                        x for x in self.ff.angles.values()
                        if type_ids == [x.id1, x.id2, x.id3]
                    ]
                    if not matches:
                        raise ValueError(
                            f"Cannot find params for angle between atom {', '.join(map(str, type_ids))}."
                        )
                    angle = matches[0]
                    self.data_fh.write(
                        f"{angle_id} {angle.id} {' '.join(map(str, [x.GetIntProp(ATOM_ID) for x in atoms]))}\n"
                    )
        self.data_fh.write(f"\n")

    def getAngleAtoms(self, atom):
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            return []
        neighbors = sorted(neighbors, key=lambda x: x.GetIntProp(TYPE_ID))
        return [[x, atom, y] for x, y in itertools.combinations(neighbors, 2)]

    def getDihedralAtoms(self, atom):
        dihe_atoms = []
        atomss = self.getAngleAtoms(atom)
        for satom, matom, eatom in atomss:
            eatomss = self.getAngleAtoms(eatom)
            matom_id = matom.GetIdx()
            eatom_id = eatom.GetIdx()
            for eatoms in eatomss:
                eatom_ids = [x.GetIdx() for x in eatoms]
                eatom_ids.remove(eatom_id)
                try:
                    eatom_ids.remove(matom_id)
                except ValueError:
                    continue
                dihe_4th = [x for x in eatoms if x.GetIdx() == eatom_ids[0]][0]
                dihe_atoms.append([satom, matom, eatom, dihe_4th])
        return dihe_atoms

    def writeDihedrals(self):
        self.data_fh.write(f"{self.DIHEDRALS.capitalize()}\n\n")
        dihedral_id = 0
        for mol in self.mols.values():
            atomss = [
                y for x in mol.GetAtoms() for y in self.getDihedralAtoms(x)
            ]
            # 1-2-3-4 and 4-3-2-1 are the same dihedral
            atomss_no_flip = []
            atom_idss = set()
            for atoms in atomss:
                atom_ids = tuple(x.GetIdx() for x in atoms)
                if atom_ids in atom_idss:
                    continue
                atom_idss.add(atom_ids)
                atom_idss.add(atom_ids[::-1])
                atomss_no_flip.append(atoms)

            for atoms in atomss_no_flip:
                dihedral_id += 1
                type_ids = [x.GetIntProp(TYPE_ID) for x in atoms]
                if type_ids[1] > type_ids[2]:
                    type_ids = type_ids[::-1]
                matches = [
                    x for x in self.ff.dihedrals.values()
                    if type_ids == [x.id1, x.id2, x.id3, x.id4]
                ]
                if not matches:
                    raise ValueError(
                        f"Cannot find params for angle between atom {', '.join(map(str, type_ids))}."
                    )
                dihedral = matches[0]
                self.data_fh.write(
                    f"{dihedral_id} {dihedral.id} {' '.join(map(str, [x.GetIntProp(ATOM_ID) for x in atoms]))}\n"
                )
        self.data_fh.write(f"\n")


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
