# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This polymer driver builds polymers from constitutional repeat units and pack
molecules into condensed phase amorphous cell.
"""
import os
import sys
import functools

from nemd import oplsua
from nemd import jobutils
from nemd import logutils
from nemd import polymutils
from nemd import structutils
from nemd import parserutils

FLAG_DENSITY = '-density'
FLAG_CELL = '-cell'
FLAG_SEED = jobutils.FLAG_SEED

GRID = 'grid'
PACK = 'pack'
GROW = 'grow'
CELL_TYPES = [GRID, PACK, GROW]

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: append time information after the message
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


class AmorphousCell(logutils.Base):
    """
    Build amorphous structure from molecules.
    """

    MINIMUM_DENSITY = 0.001

    def __init__(self, options, ff=None, **kwargs):
        """
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        :param ff 'OplsParser': the force field class.
        """
        super().__init__(**kwargs)
        self.options = options
        self.ff = ff
        self.mols = []
        self.struct = None
        if self.ff is None:
            self.ff = oplsua.get_parser(wmodel=self.options.force_field.model)

    def run(self):
        """
        Main method to build the cell.
        """
        self.setPolymers()
        self.setGriddedCell()
        self.setPackedCell()
        self.setGrowedCell()
        self.write()

    def setPolymers(self):
        """
        Build polymer from monomers if provided.
        """
        for cru, cru_num, mol_num in zip(self.options.cru,
                                         self.options.cru_num,
                                         self.options.mol_num):
            mol = polymutils.Mol(cru,
                                 cru_num,
                                 mol_num,
                                 options=self.options,
                                 logger=self.logger)
            self.mols.append(mol)

    def setGriddedCell(self):
        """
        Build gridded cell.
        """
        if self.options.cell != GRID:
            return
        self.struct = structutils.GriddedStruct.fromMols(self.mols,
                                                         ff=self.ff,
                                                         options=self.options)
        self.struct.run()

    def setPackedCell(self, mini_density=MINIMUM_DENSITY):
        """
        Build packed cell.

        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        if self.options.cell != PACK:
            return
        self.createCell(ClassStruct=structutils.PackedStruct,
                        mini_density=mini_density)

    def setGrowedCell(self, mini_density=0.01):
        """
        Build packed cell.

        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        if self.options.cell != GROW:
            return
        self.createCell(ClassStruct=structutils.GrownStruct,
                        mini_density=mini_density)

    def createCell(self,
                   ClassStruct=structutils.PackedStruct,
                   mini_density=MINIMUM_DENSITY):
        """
        Create amorphous cell.

        :param ClassStruct 'Struct': the structure class
        :param mini_density float: the minium density for liquid and solid when
            reducing it automatically.
        """
        self.struct = ClassStruct.fromMols(self.mols,
                                           ff=self.ff,
                                           options=self.options)
        density = self.options.density
        mini_density = min([mini_density, density / 5.])
        delta = min([0.1, (density - mini_density) / 4])
        while density >= mini_density:
            try:
                self.struct.runWithDensity(density)
            except structutils.DensityError:
                density -= delta if density > mini_density else mini_density
                self.log(f'Density is reduced to {density:.4f} g/cm^3')
                continue
            return

    def write(self):
        """
        Write amorphous cell into data file.
        """
        self.struct.writeData()
        for warning in self.struct.getWarnings():
            self.log_warning(f'{warning}')
        self.struct.writeIn()
        self.log(f'Data file written into {self.struct.datafile}')
        self.log(f'In script written into {self.struct.inscript}')
        jobutils.add_outfile(self.struct.datafile,
                             jobname=self.options.jobname)
        jobutils.add_outfile(self.struct.inscript,
                             jobname=self.options.jobname,
                             set_file=True)


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
    parser = polymutils.get_parser(parser=parser)
    parser.supress_arguments([polymutils.FLAG_SUBSTRUCT])
    parser.add_argument(FLAG_SEED,
                        metavar=FLAG_SEED[1:].upper(),
                        type=parserutils.type_random_seed,
                        help='Set random state using this seed.')
    parser.add_argument(
        FLAG_CELL,
        metavar=FLAG_CELL[1:].upper(),
        choices=CELL_TYPES,
        default=GROW,
        help=f'Amorphous cell type: \'{GRID}\' grids the space and '
        f'put molecules into sub-cells; \'{PACK}\' randomly '
        f'rotates and translates molecules; \'{GROW}\' grows '
        f'molecules from the smallest rigid fragments.')
    parser.add_argument(FLAG_DENSITY,
                        metavar=FLAG_DENSITY[1:].upper(),
                        type=functools.partial(parserutils.type_ranged_float,
                                               bottom=0,
                                               included_bottom=False,
                                               top=30),
                        default=0.5,
                        help=f'The density used for {PACK} and {GROW} '
                        f'amorphous cell. (g/cm^3)')
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    validator = polymutils.Validator(options)
    try:
        validator.run()
    except ValueError as err:
        parser.error(err)
    return validator.options


logger = None


def main(argv):
    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    cell = AmorphousCell(options, logger=logger)
    cell.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
