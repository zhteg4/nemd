import os
import random
import argparse
import collections
import numpy as np
from rdkit import Chem
from nemd import oplsua
from nemd import symbols
from nemd import constants
from nemd import environutils
from nemd import jobutils

FLAG_CLEAN = jobutils.FLAG_CLEAN
FLAG_JTYPE = jobutils.FLAG_JTYPE
FLAG_CPU = jobutils.FLAG_CPU
FLAG_INTERACTIVE = jobutils.FLAG_INTERACTIVE
FLAG_JOBNAME = jobutils.FLAG_JOBNAME
FLAG_DEBUG = jobutils.FLAG_DEBUG
FLAG_CPU = jobutils.FLAG_CPU
FLAG_PRJ_PATH = jobutils.FLAG_PRJ_PATH


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
        return super().add_usage(usage, actions, groups, prefix)


def get_parser(**kwargs):
    return argparse.ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, **kwargs)


def type_file(arg):
    if os.path.isfile(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} not found.')


def type_dir(arg):
    if os.path.isdir(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} is not an existing directory.')


def type_itest_dir(arg):
    try:
        return type_dir(arg)
    except argparse.ArgumentTypeError:
        dir = environutils.get_integration_test_dir()
        nargs = [arg, '0' * (4 - len(arg)) + arg]
        nargs = [os.path.join(dir, x) for x in nargs]
        for narg in nargs:
            if os.path.isdir(narg):
                return narg
    raise argparse.ArgumentTypeError(
        f"None of {', '.join([arg] + nargs)} exists.")


def type_float(arg):
    try:
        value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{arg} cannot be converted to float.')
    return value


def type_positive_float(arg):
    value = type_float(arg)
    if value <= 0:
        raise argparse.ArgumentTypeError(f'{value} is not a positive float.')
    return value


def type_ranged_float(arg,
                      bottom=-constants.LARGE_NUM,
                      top=constants.LARGE_NUM,
                      included_bottom=True,
                      include_top=True):
    value = type_float(arg)
    if included_bottom and value < bottom:
        raise argparse.ArgumentTypeError(f'{value} is smaller than {bottom}.')
    if not included_bottom and value <= bottom:
        raise argparse.ArgumentTypeError(
            f'{value} should be larger than {bottom}.')
    if include_top and value > top:
        raise argparse.ArgumentTypeError(f'{value} is larger than {bottom}.')
    if not include_top and value >= top:
        raise argparse.ArgumentTypeError(
            f'{value} should be smaller than {bottom}.')
    return value


def type_random_seed(arg):
    value = type_int(arg)
    np.random.seed(value)
    random.seed(value)
    return value


def type_int(arg):
    try:
        value = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{arg} cannot be converted to integer.')
    return value


def type_positive_int(arg):
    value = type_int(arg)
    if value < 1:
        raise argparse.ArgumentTypeError(f'{value} is not a positive integer.')
    return value


def type_smiles(arg):
    try:
        value = Chem.MolFromSmiles(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{arg} cannot be converted to integer.')
    if not value:
        raise argparse.ArgumentTypeError(f'{arg} is not a valid SMILES.')
    return value


def type_monomer_smiles(arg, allow_mol=False, canonize=True):
    value = type_smiles(arg)
    ht_atoms = [
        x for x in value.GetAtoms() if x.GetSymbol() == symbols.WILD_CARD
    ]
    if len(ht_atoms) == 0 and allow_mol:
        return Chem.CanonSmiles(arg) if canonize else arg
    if len(ht_atoms) != 2:
        raise argparse.ArgumentTypeError(
            f"{arg} doesn't contain two {symbols.WILD_CARD}.")
    for atom in ht_atoms:
        if len(atom.GetNeighbors()) != 1:
            raise argparse.ArgumentTypeError(
                f"{symbols.WILD_CARD} connects more than one atom.")
    return Chem.CanonSmiles(arg) if canonize else arg


FF_MODEL = collections.namedtuple('FF_MODEL', ['ff', 'model'])


def type_force_field(arg, ff_model=oplsua.OplsTyper.FF_MODEL):
    args = arg.split(',')
    ff_type = args[0]
    if ff_type not in ff_model:
        msg = f"Only support {','.join(ff_model.keys())}, but found {ff_type}."
        raise argparse.ArgumentTypeError(msg)
    wmodel = args[1] if len(args) == 2 else ff_model[ff_type][0]
    if wmodel not in ff_model[ff_type]:
        msg = f"Only support {','.join(ff_model[ff_type])}, but found {wmodel}."
        raise argparse.ArgumentTypeError(msg)
    return FF_MODEL(ff=ff_type, model=wmodel)


def type_slice(arg):
    args = [int(x) if x else None for x in arg.split(':')]
    if len(args) == 1:
        args += [args[0] + 1]
    if len(args) == 2:
        args += [1]
    if len(args) == 3:
        if args[0] is None:
            args[0] = 0
        if args[1] is None:
            args[1] = constants.LARGE_NUM
        if args[2] is None:
            args[2] = 1
        return args
    raise argparse.ArgumentTypeError(
        f"{arg} doesn't follow list slicing rules.")


def add_md_arguments(parser):
    parser.add_argument(oplsua.FLAG_TIMESTEP,
                        metavar='fs',
                        type=type_positive_float,
                        default=1,
                        help=f'Timestep for the MD simulation.')
    parser.add_argument(
        oplsua.FLAG_STEMP,
        metavar='K',
        type=type_positive_float,
        default=10,
        # 'Initialize the atoms with this temperature.'
        help=argparse.SUPPRESS)
    parser.add_argument(oplsua.FLAG_TEMP,
                        metavar=oplsua.FLAG_TEMP[1:].upper(),
                        type=type_positive_float,
                        default=300,
                        help=f'The equilibrium temperature target .')
    parser.add_argument(
        oplsua.FLAG_TDAMP,
        metavar=oplsua.FLAG_TDAMP[1:].upper(),
        type=type_positive_float,
        default=100,
        # Temperature damping parameter (x timestep to get the param)
        help=argparse.SUPPRESS)
    parser.add_argument(oplsua.FLAG_PRESS,
                        metavar='at',
                        type=float,
                        default=1,
                        help="The equilibrium pressure target.")
    parser.add_argument(
        oplsua.FLAG_PDAMP,
        metavar=oplsua.FLAG_PDAMP[1:].upper(),
        type=type_positive_float,
        default=1000,
        # Pressure damping parameter (x timestep to get the param)
        help=argparse.SUPPRESS)
    parser.add_argument(
        oplsua.FLAG_LJ_CUT,
        metavar=oplsua.FLAG_LJ_CUT[1:].upper(),
        type=type_positive_float,
        default=oplsua.LammpsIn.DEFAULT_LJ_CUT,
        # Cut off for the lennard jones
        help=argparse.SUPPRESS)
    parser.add_argument(
        oplsua.FLAG_COUL_CUT,
        metavar=oplsua.FLAG_COUL_CUT[1:].upper(),
        type=type_positive_float,
        default=oplsua.LammpsIn.DEFAULT_COUL_CUT,
        # Cut off for the coulombic interaction
        help=argparse.SUPPRESS)
    parser.add_argument(oplsua.FLAG_RELAX_TIME,
                        metavar='ns',
                        type=type_positive_float,
                        default=4,
                        help='Relaxation simulation time.')
    parser.add_argument(oplsua.FLAG_PROD_TIME,
                        metavar='ns',
                        type=type_positive_float,
                        default=4,
                        help='Production simulation time.')
    parser.add_argument(oplsua.FLAG_PROD_ENS,
                        metavar=oplsua.FLAG_PROD_ENS[1:].upper(),
                        choices=oplsua.ENSEMBLES,
                        default=oplsua.NVE,
                        help='Production ensemble.')
    parser.add_argument(
        oplsua.FlAG_FORCE_FIELD,
        metavar=oplsua.FlAG_FORCE_FIELD[1:].upper(),
        type=type_force_field,
        default=oplsua.OplsTyper.OPLSUA_TIP3P,
        help='The force field type (and water model separated with comma).')


def add_job_arguments(parser, arg_flags=None, jobname=None):
    """
    Add job control related flags.

    :param parser: the parser to add arguments
    :type parser: 'argparse.ArgumentParser'
    :param arg_flags: specific job control related flags to add
    :type arg_flags: list
    :param jobname: the default jobname
    :type jobname: str
    """
    if arg_flags is None:
        arg_flags = [FLAG_INTERACTIVE, FLAG_JOBNAME, FLAG_DEBUG, FLAG_CPU]
    # Workflow drivers may add the job control options a few times
    if FLAG_JOBNAME in arg_flags and FLAG_JOBNAME in parser._option_string_actions:
        parser.set_defaults(jobname=jobname)
    arg_flags = [
        x for x in arg_flags if x not in parser._option_string_actions
    ]
    if FLAG_INTERACTIVE in arg_flags:
        parser.add_argument(FLAG_INTERACTIVE,
                            dest=FLAG_INTERACTIVE[1:].lower(),
                            action='store_true',
                            help='Enable interactive mode')
    if FLAG_JOBNAME in arg_flags:
        parser.add_argument(
            FLAG_JOBNAME,
            dest=FLAG_JOBNAME[1:].lower(),
            default=jobname,
            help='The jobname based on which filenames are created.')
    if FLAG_DEBUG in arg_flags:
        parser.add_argument(
            FLAG_DEBUG,
            action='store_true',
            dest=FLAG_DEBUG[1:].lower(),
            help='Enable debug mode (e.g. extra printing and files)')
    if FLAG_CPU in arg_flags:
        parser.add_argument(FLAG_CPU,
                            type=type_positive_int,
                            dest=FLAG_CPU[1:].lower(),
                            default=round(os.cpu_count() / 2),
                            help='Number of CPU processors.')


def add_workflow_arguments(parser, arg_flags=None):
    """
    Add workflow related flags.

    :param parser: the parser to add arguments
    :type parser: 'argparse.ArgumentParser'
    :param arg_flags: specific workflow related flags to add
    :type arg_flags: list
    """
    if arg_flags is None:
        arg_flags = [FLAG_CLEAN, FLAG_JTYPE, FLAG_CPU]
    if FLAG_CLEAN in arg_flags:
        parser.add_argument(
            FLAG_CLEAN,
            action='store_true',
            help='Clean previous workflow results (if any) and run new ones.')
    if FLAG_JTYPE in arg_flags:
        parser.add_argument(
            FLAG_JTYPE,
            choices=[jobutils.TASK, jobutils.AGGREGATOR],
            default=[jobutils.TASK, jobutils.AGGREGATOR],
            help=f'{jobutils.TASK} jobs run tasks and each task has to register '
            f'one outfile to be considered as completed; '
            f'{jobutils.AGGREGATOR} jobs run after the all task jobs '
            f'finish.')
        parser.add_argument(
            FLAG_PRJ_PATH,
            default=os.curdir,
            type=type_dir,
            help=
            f'Project path if only {jobutils.AGGREGATOR} jobs are requested.')
