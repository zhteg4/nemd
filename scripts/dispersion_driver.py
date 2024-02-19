# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This driver to perform phonon calculations.
"""
import os
import sys
import scipy
import crystals
import itertools

import numpy as np
from rdkit import Chem
from scipy import constants
from nemd import symbols
from nemd import jobutils
from nemd import logutils
from nemd import oplsua
from nemd import parserutils
from nemd import environutils

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')


def log_debug(msg):
    """
    Print this message into the log file in debug mode.

    :param msg str: the msg to be printed
    """
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: print time after the msg
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


class AlaWriter(object):
    """
    """
    AND = symbols.AND
    FORWARDSLASH = symbols.FORWARDSLASH

    GENERAL = 'general'
    PREFIX = 'PREFIX'
    MODE = 'MODE'
    NAT = 'NAT'
    NKD = 'NKD'
    KD = 'KD'

    OPTIMIZE = 'optimize'
    DFSET = 'DFSET'

    SUGGEST = 'suggest'
    INTERACTION = 'interaction'
    NORDER = 'NORDER'

    CELL = 'cell'
    CUTOFF = 'cutoff'
    POSITION = 'position'

    def __init__(self, filename, mode=SUGGEST):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.filename = filename
        self.mode = mode
        self.data = {}

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setSuperCell()
        self.setGeneral()
        self.setInteraction()
        self.setCell()
        self.setCutoff()
        self.setPosition()
        self.write()

    def setSuperCell(self):
        ucell_xtal = crystals.Crystal.from_database('Si')
        self.elements = list(ucell_xtal.chemical_composition.keys())
        self.scell = ucell_xtal.supercell(2, 2, 2)

    def setGeneral(self):
        prefix = 'si222'
        nat = len(self.scell.atoms)
        nkd = len(self.elements)
        kd = ','.join(self.elements)
        general = [
            f"{self.PREFIX} = {prefix}", f"{self.MODE} = {self.mode}",
            f"{self.NAT} = {nat}", f"{self.NKD} = {nkd}", f"{self.KD} = {kd}"
        ]
        self.data[self.GENERAL] = general

    def setOptimize(self):
        if self.mode != self.OPTIMIZE:
            return
        self.data[self.DFSET] = f"{self.DFSET}_harmonic"

    def setInteraction(self):
        norder = 1  # 1: harmonic, 2: cubic, ..
        self.data[self.INTERACTION] = [f"{self.NORDER} = {norder}"]

    def setCell(self):
        bohr_radius = scipy.constants.physical_constants['Bohr radius'][0]
        angstrom = scipy.constants.angstrom
        scale = angstrom / bohr_radius  # Bohr unit
        vectors = [
            x * y
            for x, y in zip(self.scell.lattice_vectors, self.scell.dimensions)
        ]
        self.vectors = [x * scale for x in vectors]
        cell = ["1"] + [self.pos_fmt(x) for x in self.vectors]
        self.data[self.CELL] = cell

    def pos_fmt(self, numbers):
        return ' '.join(map('{:.8f}'.format, numbers))

    def setCutoff(self):
        pairs = itertools.combinations_with_replacement(self.elements, 2)
        self.data[self.CUTOFF] = [f"{x}-{y} 7.3" for x, y in pairs]

    def setPosition(self):
        atoms = [x for x in self.scell.atoms]
        atoms = sorted(atoms, key=lambda x: tuple(x.coords_fractional))
        pos = [[x.element, x.coords_fractional / self.scell.dimensions]
               for x in atoms]
        pos = [
            f"{self.elements.index(e) + 1} {self.pos_fmt(x)}" for e, x in pos
        ]
        self.data[self.POSITION] = pos

    def write(self):
        with open(self.filename, 'w') as fh:
            for key, val in self.data.items():
                if not val:
                    continue
                fh.write(f"{self.AND}{key}\n")
                for line in val:
                    fh.write(f"  {line}\n")
                fh.write(f"{self.FORWARDSLASH}\n")
                fh.write("\n")

    def getMol(self):
        atoms = [x for x in self.scell.atoms]
        atoms = sorted(atoms, key=lambda x: tuple(x.coords_fractional))
        mol = Chem.Mol()
        emol = Chem.EditableMol(mol)
        idxs = [emol.AddAtom(Chem.rdchem.Atom(x.atomic_number)) for x in atoms]
        mol = emol.GetMol()
        cfm = Chem.rdchem.Conformer(mol.GetNumAtoms())
        [cfm.SetAtomPosition(x, atoms[x].coords_cartesian) for x in idxs]
        mol.AddConformer(cfm)
        return Mol(mol,
                   lattice_parameters=self.scell.lattice_parameters,
                   dimensions=self.scell.dimensions)


class LammpsData(oplsua.LammpsData):

    def writeData(self, adjust_coords=False):

        with open(self.lammps_data, 'w') as self.data_fh:
            self.setAtoms()
            self.writeDescription()
            self.writeTopoType()
            self.writeBox()
            self.writeMasses()
            self.writeAtoms()

    def setAtoms(self):
        super().setAtoms()
        elements = [
            y.GetAtomicNum() for x in self.mols.values() for y in x.GetAtoms()
        ]
        self.elements = list(set(elements))

    def writeDescription(self):
        """
        Write the lammps description section, including the number of atom, bond,
        angle etc.
        """
        if self.mols is None:
            raise ValueError(f"Mols are not set.")

        self.data_fh.write(f"{self.LAMMPS_DESCRIPTION}\n\n")
        atom_nums = [len(x.GetAtoms()) for x in self.mols.values()]
        self.data_fh.write(f"{sum(atom_nums)} {self.ATOMS}\n\n")

    def writeTopoType(self):
        """
        Write topologic data. e.g. number of atoms, angles...
        """
        self.data_fh.write(f"{len(self.elements)} {self.ATOM_TYPES}\n\n")

    def writeBox(self, min_box=None, buffer=None):
        """
        Write box information.

        :param min_box list: minimum box size
        :param buffer list: buffer in three dimensions
        """

        if self.mols and bool(self.mols[1].lattice_parameters):
            param = self.mols[1].lattice_parameters[:3]
            boxes = np.array(param) * self.mols[1].dimensions
            for dim in range(3):
                self.data_fh.write(
                    f"{0:.4f} {boxes[dim]:.4f} {self.LO_HI[dim]}\n")
            # FIXME https://docs.lammps.org/Howto_triclinic.html
            self.data_fh.write("0.0000 0.0000 0.0000 xy xz yz\n")
            self.data_fh.write("\n")
            return
        super().writeBox()

    def writeMasses(self):
        """
        Write out mass information.
        """
        self.data_fh.write(f"{self.MASSES}\n\n")
        masses = list(
            set([
                y.GetMass() for x in self.mols.values() for y in x.GetAtoms()
            ]))
        for id, mass in enumerate(masses, 1):
            self.data_fh.write(f"{id} {mass}\n")
        self.data_fh.write(f"\n")

    def writeAtoms(self):
        """
        Write atom coefficients.

        :param comments bool: If True, additional descriptions including element
            sysmbol are written after each atom line
        """

        self.data_fh.write(f"{self.ATOMS.capitalize()}\n\n")
        for mol_id, mol in self.mols.items():
            data = np.zeros((mol.GetNumAtoms(), 5))
            conformer = mol.GetConformer()
            data[:, 0] = [x.GetIntProp(self.ATOM_ID) for x in mol.GetAtoms()]
            data[:, 1] = mol_id
            data[:, 2:] = conformer.GetPositions()
            np.savetxt(self.data_fh, data, fmt='%i %i %.3f %.3f %.3f')
        self.data_fh.write(f"\n")


class Mol(Chem.rdchem.Mol):

    LATTICE_PARAMETERS = 'lattice_parameters'
    DIMENSIONS = 'dimensions'

    def __init__(self, *args, **kwargs):
        self.lattice_parameters = kwargs.pop(self.LATTICE_PARAMETERS, None)
        self.dimensions = kwargs.pop(self.DIMENSIONS, (
            1,
            1,
            1,
        ))
        super().__init__(*args, **kwargs)


class Dispersion(object):

    IN = '.in'
    DSP_IN = f'_dsp{IN}'

    SUBMODULE_PATH = environutils.get_submodule_path()
    ALAMODE = environutils.ALAMODE
    ALAMODE_SRC = os.path.join(SUBMODULE_PATH, ALAMODE, ALAMODE)
    Si_LAMMPS = os.path.join(ALAMODE_SRC, 'example', 'Si_LAMMPS')

    def __init__(self, options):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.options = options
        # FIXME: should read in timestep to calculate the time
        self.dsp_in = self.options.jobname + self.DSP_IN

    def run(self):
        ala_writer = AlaWriter(self.dsp_in)
        ala_writer.run()
        mol = ala_writer.getMol()
        LammpsData({1: mol}, oplsua.get_opls_parser(), 'wa').writeData()

    @classmethod
    def getOutfiles(cls, logfile):
        """
        Get the output files based on the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :return: task name and the related outfile
        :rtype: dict
        """
        jobname = cls.getLogged(logfile)[0]
        tsks = cls.getLogged(logfile, key=cls.TASK, strip='[]', delimiter=', ')
        return {x: jobname + cls.DATA_EXT % x for x in tsks}

    @classmethod
    def getLogged(cls, logfile, key=None, strip=None, delimiter=None):
        """
        Get the values corresponding to the key in the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :param key: the key based on which values are fetched
        :type key: str
        :param delimiter: the chars to strip the string
        :type delimiter: str
        :param delimiter: the delimiter to split the string
        :type delimiter: str
        :return: the matching values in the logfile
        :rtype: list
        """
        pass

    @classmethod
    def combine(cls, files, log, name, inav=False):
        """
        Combine multiple outfiles from the same task into one.

        :param files: task name and the related outfile
        :type files: dict
        :param log: the function to print user-facing information
        :type log: 'function'
        :param name: output files are named based on this name
        :type name: str
        :param inav: pop up window and show plot during code execution if
            interactive mode is on
        :type inav: bool
        """
        pass


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
    parserutils.add_job_arguments(parser,
                                  jobname=environutils.get_jobname(JOBNAME))
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    return options


logger = None


def main(argv):

    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    dispersion = Dispersion(options)
    dispersion.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
