# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This module provides utilities for alamode submodule.
"""
import sh
import scipy
import itertools
from nemd import symbols
from scipy import constants


class AlaWriter(object):
    """
    Write the input script for the alamode submodule.
    """

    IN = '.in'
    XML_EXT = '.xml'

    AND = symbols.AND
    FORWARDSLASH = symbols.FORWARDSLASH

    GENERAL = 'general'
    PREFIX = 'PREFIX'
    MODE = 'MODE'
    NAT = 'NAT'
    NKD = 'NKD'
    KD = 'KD'
    MASS = 'MASS'

    OPTIMIZE = 'optimize'
    DFSET = 'dfset'

    SUGGEST = 'suggest'
    PAT = 'pat'

    PHONONS = 'phonons'
    FCSXML = 'FCSXML'
    PH = 'ph'

    INTERACTION = 'interaction'
    NORDER = 'NORDER'

    CELL = 'cell'
    CUTOFF = 'cutoff'
    POSITION = 'position'

    KPOINT = 'kpoint'

    EXT = {OPTIMIZE: DFSET, SUGGEST: PAT, PHONONS: PH}

    # https://en.wikipedia.org/wiki/Brillouin_zone
    # https://www.businessballs.com/glossaries-and-terminology/greek-alphabet/
    # Center of the Brillouin zone
    GAMMA = 'G'
    GAMMA_PNT = '0 0 0'
    # Simple cube
    # Center of a face
    CHI = 'X'
    CHI_PNT = '0.5 0.5 0.0'
    # Corner point
    RHO = 'R'
    RHO_PNT = '0.5 0.5 1'
    # Center of a hexagonal face
    LAMBDA = 'L'
    LAMBDA_PNT = '0.5 0.5 0.5'

    def __init__(self, scell, jobname=None, mode=SUGGEST):
        """
        :param scell 'crystals.crystal.Supercell': the crystal cell
        :param jobname str: the jobname
        :param mode str: the mode of the calculation
        """
        self.cell = scell
        self.jobname = jobname
        self.elements = list(self.cell.chemical_composition.keys())
        self.mode = mode
        self.data = {}
        if self.jobname is None:
            dimensions = 'x'.join(map(str, self.cell.dimensions))
            self.jobname = f"{self.cell.chemical_formula}_{dimensions}"
        self.filename = f"{self.jobname}_{self.EXT[self.mode]}{self.IN}"

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setGeneral()
        self.setOptimize()
        self.setInteraction()
        self.setCell()
        self.setCutoff()
        self.setPosition()
        self.setKpoint()
        self.write()

    def setGeneral(self):
        """
        Set the &general-filed.

        https://alamode.readthedocs.io/en/latest/almdir/inputalm.html#general-field
        PREFIX sets the output file names
        NAT sets the total atoms in the supercell
        NKD sets the atom specie number
        KD sets the atom specie names
        """
        nat = len(self.cell.atoms)
        nkd = len(self.elements)
        kd = ','.join(self.elements)
        general = [
            f"{self.PREFIX} = {self.jobname}", f"{self.MODE} = {self.mode}"
        ]
        if self.mode in [self.SUGGEST, self.OPTIMIZE]:
            general += [f"{self.NAT} = {nat}"]
        if self.mode == self.PHONONS:
            general += [f"{self.FCSXML} = {self.jobname}{self.XML_EXT}"]
        general += [f"{self.NKD} = {nkd}", f"{self.KD} = {kd}"]
        if self.mode == self.PHONONS:
            unitcell = [x for x in self.cell.unitcell][0]
            general += [f"{self.MASS} = {unitcell.mass}"]
        self.data[self.GENERAL] = general

    def setOptimize(self):
        """
        Set the &optimize-filed.

        https://alamode.readthedocs.io/en/latest/almdir/inputalm.html#optimize-field
        DFSET sets the filename containing the displacement-force datset.
        """
        if self.mode != self.OPTIMIZE:
            return
        self.data[self.OPTIMIZE] = [
            f"{self.DFSET.upper()} = {self.jobname}.{self.DFSET}_harmonic"
        ]

    def setInteraction(self):
        """
        Set the &interaction-field.

        https://alamode.readthedocs.io/en/latest/almdir/inputalm.html#interaction-field
        NORDER set the order of force constant.
        """
        if self.mode not in [self.SUGGEST, self.OPTIMIZE]:
            return
        norder = 1  # 1: harmonic, 2: cubic, ..
        self.data[self.INTERACTION] = [f"{self.NORDER} = {norder}"]

    def setCell(self):
        """
        Set the &cell-field: cell parameters.

        https://alamode.readthedocs.io/en/latest/almdir/inputalm.html#interaction-field
        """
        bohr_radius = scipy.constants.physical_constants['Bohr radius'][0]
        angstrom = scipy.constants.angstrom
        scale = angstrom / bohr_radius  # Bohr unit
        vectors = self.cell.lattice_vectors
        if self.mode in [self.SUGGEST, self.OPTIMIZE]:
            vectors = [x * y for x, y in zip(vectors, self.cell.dimensions)]
        self.vectors = [x * scale for x in vectors]
        cell = ["1"] + [self.pos_fmt(x) for x in self.vectors]
        self.data[self.CELL] = cell

    def pos_fmt(self, numbers):
        """
        Format the position format.

        :param numbers float: input float
        :return str: formatted str
        """
        return ' '.join(map('{:.8f}'.format, numbers))

    def setCutoff(self):
        """
        Set the cutoff for neighbor searching.

        https://alamode.readthedocs.io/en/latest/almdir/inputalm.html#cutoff-field
        """
        if self.mode not in [self.SUGGEST, self.OPTIMIZE]:
            return
        pairs = itertools.combinations_with_replacement(self.elements, 2)
        self.data[self.CUTOFF] = [f"{x}-{y} 7.3" for x, y in pairs]

    def setPosition(self):
        """
        Set the &position-field for atom species and coordinates.

        https://alamode.readthedocs.io/en/latest/almdir/inputalm.html#position-field
        """
        if self.mode not in [self.SUGGEST, self.OPTIMIZE]:
            return
        atoms = [x for x in self.cell.atoms]
        atoms = sorted(atoms, key=lambda x: tuple(x.coords_fractional))
        pos = [[x.element, x.coords_fractional / self.cell.dimensions]
               for x in atoms]
        pos = [
            f"{self.elements.index(e) + 1} {self.pos_fmt(x)}" for e, x in pos
        ]
        self.data[self.POSITION] = pos

    def setKpoint(self, num=51):
        """
        Set the &kpoint-field

        :param num int: number of grid between two points.
        https://alamode.readthedocs.io/en/latest/anphondir/inputanphon.html#kpoint-field
        """

        if self.mode != self.PHONONS:
            return
        self.data[self.KPOINT] = ['1 # line mode']
        lines = [
            f"{self.GAMMA} {self.GAMMA_PNT} {self.CHI} {self.CHI_PNT} {num}"
        ]
        lines += [
            f"{self.RHO} {self.RHO_PNT} {self.GAMMA} {self.GAMMA_PNT} {num}"
        ]
        lines += [
            f"{self.GAMMA} {self.GAMMA_PNT} {self.LAMBDA} {self.LAMBDA_PNT} {num}"
        ]
        self.data[self.KPOINT] += lines

    def write(self):
        """
        Write out the alamode input script.
        """
        with open(self.filename, 'w') as fh:
            for key, val in self.data.items():
                if not val:
                    continue
                fh.write(f"{self.AND}{key}\n")
                for line in val:
                    fh.write(f"  {line}\n")
                fh.write(f"{self.FORWARDSLASH}\n")
                fh.write("\n")


class AlaLogReader(object):

    SYMMETRY = 'SYMMETRY'
    SUGGESTED_DSIP_FILE = 'Suggested displacement patterns are printed in ' \
                          'the following files:'
    INPUT_FOR_ANPHON = 'Input data for the phonon code ANPHON      :'
    PHONON_BAND_STRUCTURE = "Phonon band structure"
    PHONON_BAND_STRUCTURE_EXT = ": " + PHONON_BAND_STRUCTURE

    def __init__(self, filename):
        """
        :param filename str: The logging filename generated by alamode.
        """
        self.filename = filename
        self.disp_pattern_file = None

    def getDispPatternFile(self):
        """
        Get the filename for suggested displacements

        :return str: the filename for suggested displacements.
        """
        lines = sh.grep(self.SUGGESTED_DSIP_FILE, self.filename, '-A 1')
        return lines.split()[-1]

    def getAfcsXml(self):
        """
        Get the filename for suggested displacements

        :return str: the filename for suggested displacements.
        """
        lines = sh.grep(self.INPUT_FOR_ANPHON, self.filename, '-A 1')
        return lines.split()[-1]

    def getPhBands(self):
        """
        Get the filename for suggested displacements

        :return str: the filename for suggested displacements.
        """
        lines = sh.grep(self.PHONON_BAND_STRUCTURE_EXT, self.filename)
        return lines.split()[0]
