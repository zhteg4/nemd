# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module adds jobcontrol related command line flags.
"""
import argparse
from nemd import oplsua
from nemd import parserutils

FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
FLAG_DEBUG = '-DEBUG'

FLAG_TIMESTEP = '-timestep'
FLAG_STEMP = '-stemp'
FLAG_TEMP = '-temp'
FLAG_TDAMP = '-tdamp'
FLAG_PRESS = '-press'
FLAG_PDAMP = '-pdamp'
FLAG_LJ_CUT = '-lj_cut'
FLAG_COUL_CUT = '-coul_cut'
FlAG_FORCE_FIELD = '-force_field'


def add_job_arguments(parser, arg_flags=None):
    if arg_flags is None:
        arg_flags = [FLAG_INTERACTIVE, FLAG_JOBNAME, FLAG_DEBUG]

    if FLAG_INTERACTIVE in arg_flags:
        parser.add_argument(FLAG_INTERACTIVE,
                            dest=FLAG_INTERACTIVE[1:].lower(),
                            action='store_true',
                            help='')
    if FLAG_JOBNAME in arg_flags:
        parser.add_argument(FLAG_JOBNAME,
                            dest=FLAG_JOBNAME[1:].lower(),
                            help='')
    if FLAG_DEBUG in arg_flags:
        parser.add_argument(FLAG_DEBUG,
                            action='store_true',
                            dest=FLAG_DEBUG[1:].lower(),
                            help='')


def add_md_arguments(parser):
    parser.add_argument(FLAG_TIMESTEP,
                        metavar='fs',
                        type=parserutils.type_positive_float,
                        default=1,
                        help=f'Timestep for the MD simulation.')
    parser.add_argument(
        FLAG_STEMP,
        metavar='K',
        type=parserutils.type_positive_float,
        default=10,
        # 'Initialize the atoms with this temperature.'
        help=argparse.SUPPRESS)
    parser.add_argument(FLAG_TEMP,
                        metavar=FLAG_TEMP[1:].upper(),
                        type=parserutils.type_positive_float,
                        default=300,
                        help=f'The equilibrium temperature target .')
    parser.add_argument(
        FLAG_TDAMP,
        metavar=FLAG_TDAMP[1:].upper(),
        type=parserutils.type_positive_float,
        default=100,
        # Temperature damping parameter (x timestep to get the param)
        help=argparse.SUPPRESS)
    parser.add_argument(FLAG_PRESS,
                        metavar='at',
                        type=float,
                        default=1,
                        help="The equilibrium pressure target.")
    parser.add_argument(
        FLAG_PDAMP,
        metavar=FLAG_PDAMP[1:].upper(),
        type=parserutils.type_positive_float,
        default=1000,
        # Pressure damping parameter (x timestep to get the param)
        help=argparse.SUPPRESS)
    parser.add_argument(
        FLAG_LJ_CUT,
        metavar=FLAG_LJ_CUT[1:].upper(),
        type=parserutils.type_positive_float,
        default=11.,
        # Cut off for the lennard jones
        help=argparse.SUPPRESS)
    parser.add_argument(
        FLAG_COUL_CUT,
        metavar=FLAG_COUL_CUT[1:].upper(),
        type=parserutils.type_positive_float,
        default=11.,
        # Cut off for the coulombic interaction
        help=argparse.SUPPRESS)
    parser.add_argument(
        FlAG_FORCE_FIELD,
        metavar=FlAG_FORCE_FIELD[1:].upper(),
        type=parserutils.type_force_field,
        default=oplsua.OplsTyper.OPLSUA_TIP3P,
        help='The force field type (and model for water separated with comma).'
    )
