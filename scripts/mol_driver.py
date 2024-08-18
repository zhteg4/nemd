import os
import sys

from nemd import oplsua
from nemd import jobutils
from nemd import logutils
from nemd import lammpsin
from nemd import lammpsfix
from nemd import polymutils
from nemd import structutils
from nemd import parserutils

FLAG_DEFAULTS = {
    polymutils.FLAG_NO_MINIMIZE: True,
    polymutils.FLAG_BUFFER: f"{lammpsin.In.DEFAULT_CUT * 4}",
    polymutils.FLAG_MOL_NUM: [1],
    parserutils.FLAG_TEMP: 0,
    parserutils.FLAG_TIMESTEP: 1,
    parserutils.FLAG_PRESS: 1,
    parserutils.FLAG_RELAX_TIME: 0,
    parserutils.FLAG_PROD_TIME: 0,
    parserutils.FLAG_PROD_ENS: lammpsfix.NVE
}

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


class GridCell:
    """
    A grid cell with a fixed number of molecules.
    """

    def __init__(self, options, ff=None):
        """
        :param options 'argparse.ArgumentParser':  Parsed command-line options
        :param ff 'OplsParser': the force field class.
        """
        self.options = options
        self.ff = ff
        self.mol = None
        self.struct = None
        if self.ff is None:
            self.ff = oplsua.get_parser(wmodel=self.options.force_field.model)

    def run(self):
        """
        Main method to build the cell.
        """
        self.setPolymer()
        self.setCell()
        self.write()

    def setPolymer(self):
        """
        Build polymer from monomers if provided.
        """
        self.mol = polymutils.Mol(self.options.cru[0],
                                  self.options.cru_num[0],
                                  self.options.mol_num[0],
                                  options=self.options,
                                  logger=logger)

    def setCell(self):
        """
        Build gridded cell.
        """
        self.struct = structutils.GriddedStruct.fromMols([self.mol],
                                                         ff=self.ff,
                                                         options=self.options)
        self.struct.run()

    def write(self):
        """
        Write amorphous cell into data file.
        """
        self.struct.writeData()
        self.struct.writeIn()
        log(f'Data file written into {self.struct.datafile}')
        log(f'In script written into {self.struct.inscript}')
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
    parser.set_defaults(**{x[1:]: y for x, y in FLAG_DEFAULTS.items()})
    parser = polymutils.add_arguments(parser)
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
    parser.supress_arguments(FLAG_DEFAULTS.keys())
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
    cell = GridCell(options, logger=logger)
    cell.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
