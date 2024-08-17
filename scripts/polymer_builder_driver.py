# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This polymer driver builds polymers from constitutional repeat units and pack
molecules into condensed phase amorphous cell.

'mpirun -np 4 lmp_mpi -in polymer_builder.in' runs with 4 processors
'lmp_serial -in polymer_builder.in' runs with 1 processor
"""
import os
import sys

from nemd import jobutils
from nemd import logutils
from nemd import polymutils
from nemd import parserutils

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
    parser.supress_arguments([polymutils.FLAG_SUBSTRUCT])
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
    cell = polymutils.AmorphousCell(options, logger=logger)
    cell.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
