import os
import sys

from nemd import jobutils
from nemd import logutils
from nemd import lammpsin
from nemd import lammpsfix
from nemd import polymutils
from nemd import parserutils

FLAG_DEFAULTS = {
    polymutils.FLAG_NO_MINIMIZE: True,
    polymutils.FLAG_CELL: polymutils.GRID,
    polymutils.FLAG_BUFFER: f"{lammpsin.In.DEFAULT_CUT * 4}",
    polymutils.FLAG_DENSITY: 1,
    polymutils.FLAG_MOL_NUM: [1],
    parserutils.FLAG_TEMP: 0,
    parserutils.FLAG_TIMESTEP: 1,
    parserutils.FLAG_PRESS: 1,
    parserutils.FLAG_RELAX_TIME: 0,
    parserutils.FLAG_PROD_TIME: 0,
    parserutils.FLAG_PROD_ENS: lammpsfix.NVE,
    jobutils.FLAG_SEED: 0
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


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """

    parser = polymutils.get_parser(parser=parser)
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
    parser.set_defaults(**{x[1:]: y for x, y in FLAG_DEFAULTS.items()})
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
    cell = polymutils.AmorphousCell(options, logger=logger)
    cell.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
