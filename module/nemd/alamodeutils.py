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

    def __init__(self, scell, jobname=None, mode=SUGGEST):
        """
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
        if self.mode != self.OPTIMIZE:
            return
        self.data[self.OPTIMIZE] = [
            f"{self.DFSET.upper()} = {self.jobname}.{self.DFSET}_harmonic"
        ]

    def setInteraction(self):
        if self.mode not in [self.SUGGEST, self.OPTIMIZE]:
            return
        norder = 1  # 1: harmonic, 2: cubic, ..
        self.data[self.INTERACTION] = [f"{self.NORDER} = {norder}"]

    def setCell(self):
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
        return ' '.join(map('{:.8f}'.format, numbers))

    def setCutoff(self):
        if self.mode not in [self.SUGGEST, self.OPTIMIZE]:
            return
        pairs = itertools.combinations_with_replacement(self.elements, 2)
        self.data[self.CUTOFF] = [f"{x}-{y} 7.3" for x, y in pairs]

    def setPosition(self):
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

    def setKpoint(self):
        if self.mode != self.PHONONS:
            return
        self.data[self.KPOINT] = ['1\n # line mode']
        import pdb
        pdb.set_trace()

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
