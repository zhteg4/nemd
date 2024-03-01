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
    """
    IN = '.in'
    PAT = 'pat'

    AND = symbols.AND
    FORWARDSLASH = symbols.FORWARDSLASH

    GENERAL = 'general'
    PREFIX = 'PREFIX'
    MODE = 'MODE'
    NAT = 'NAT'
    NKD = 'NKD'
    KD = 'KD'

    OPTIMIZE = 'optimize'
    DFSET = 'dfset'

    SUGGEST = 'suggest'
    INTERACTION = 'interaction'
    NORDER = 'NORDER'

    CELL = 'cell'
    CUTOFF = 'cutoff'
    POSITION = 'position'

    def __init__(self, scell, jobname=None, mode=SUGGEST):
        """
        """
        self.scell = scell
        self.jobname = jobname
        self.elements = list(self.scell.chemical_composition.keys())
        self.mode = mode
        self.data = {}
        mode = self.PAT if mode == self.SUGGEST else self.DFSET
        if self.jobname is None:
            dimensions = 'x'.join(map(str, self.scell.dimensions))
            self.jobname = f"{self.scell.chemical_formula}_{dimensions}"
        self.filename = f"{self.jobname}_{mode}{self.IN}"

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
        self.write()

    def setGeneral(self):
        nat = len(self.scell.atoms)
        nkd = len(self.elements)
        kd = ','.join(self.elements)
        general = [
            f"{self.PREFIX} = {self.jobname}", f"{self.MODE} = {self.mode}",
            f"{self.NAT} = {nat}", f"{self.NKD} = {nkd}", f"{self.KD} = {kd}"
        ]
        self.data[self.GENERAL] = general

    def setOptimize(self):
        if self.mode != self.OPTIMIZE:
            return
        self.data[self.OPTIMIZE] = [
            f"{self.DFSET.upper()} = {self.jobname}.{self.DFSET}_harmonic"
        ]

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


class AlaLogReader(object):

    SYMMETRY = 'SYMMETRY'
    SUGGESTED_DSIP_FILE = 'Suggested displacement patterns are printed in ' \
                          'the following files:'
    INPUT_FOR_ANPHON = 'Input data for the phonon code ANPHON      :'

    def __init__(self, filename):
        self.filename = filename
        self.disp_pattern_file = None

    def run(self):
        self.disp_pattern_file = self.getDispPatternFile()
        self.afcs_xml = self.getAfcsXml()

    def getDispPatternFile(self):
        lines = sh.grep(self.SUGGESTED_DSIP_FILE, self.filename, '-A 1')
        return lines.split()[-1]

    def getAfcsXml(self):
        lines = sh.grep(self.INPUT_FOR_ANPHON, self.filename, '-A 1')
        return lines.split()[-1]
