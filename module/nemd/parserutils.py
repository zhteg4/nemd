import os
import random
import argparse
import collections
import numpy as np
from rdkit import Chem
from nemd import symbols
from nemd import lammpsin
from nemd import lammpsfix
from nemd import constants
from nemd import jobutils
from nemd import environutils

FLAG_STATE_NUM = jobutils.FLAG_STATE_NUM
FLAG_CLEAN = jobutils.FLAG_CLEAN
FLAG_JTYPE = jobutils.FLAG_JTYPE
FLAG_CPU = jobutils.FLAG_CPU
FLAG_INTERACTIVE = jobutils.FLAG_INTERACTIVE
FLAG_JOBNAME = jobutils.FLAG_JOBNAME
FLAG_DEBUG = jobutils.FLAG_DEBUG
FLAG_PYTHON = jobutils.FLAG_PYTHON
FLAG_CPU = jobutils.FLAG_CPU
FLAG_PRJ_PATH = jobutils.FLAG_PRJ_PATH
FLAG_SEED = jobutils.FLAG_SEED

DEFAULT_JOB_FLAGS = [
    FLAG_INTERACTIVE, FLAG_JOBNAME, FLAG_DEBUG, FLAG_PYTHON, FLAG_CPU
]
DEFAULT_WORKFLOW_FLAGS = [FLAG_STATE_NUM, FLAG_CLEAN, FLAG_JTYPE, FLAG_CPU]

FLAG_TIMESTEP = '-timestep'
FLAG_STEMP = '-stemp'
FLAG_TEMP = '-temp'
FLAG_TDAMP = '-tdamp'
FLAG_PRESS = '-press'
FLAG_PDAMP = '-pdamp'
FLAG_LJ_CUT = '-lj_cut'
FLAG_COUL_CUT = '-coul_cut'
FLAG_RELAX_TIME = '-relax_time'
FLAG_PROD_TIME = '-prod_time'
FLAG_PROD_ENS = '-prod_ens'
FlAG_FORCE_FIELD = '-force_field'

FLAG_TRAJ = 'traj'
FLAG_DATA_FILE = '-data_file'
FLAG_LAST_PCT = '-last_pct'
FLAG_SLICE = '-slice'


class ArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
        return super().add_usage(usage, actions, groups, prefix)


class ArgumentParser(argparse.ArgumentParser):

    def supress_arguments(self, to_supress):
        """
        Supress the help messages of specified arguments.

        :param parser: the parser to add arguments
        :type parser: 'argparse.ArgumentParser'
        :param to_supress: the arguments to be suppressed
        :type to_supress: set
        """
        to_supress = set(to_supress)
        for action in self._actions:
            if to_supress.intersection(action.option_strings):
                action.help = argparse.SUPPRESS


def get_parser(**kwargs):
    return ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                          **kwargs)


def type_file(arg):
    if os.path.isfile(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} not found.')


def type_dir(arg):
    if os.path.isdir(arg):
        return arg
    raise argparse.ArgumentTypeError(f'{arg} is not an existing directory.')


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


def type_nonnegative_float(arg):
    value = type_float(arg)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f'{value} is not a nonnegative float.')
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
            f'{value} should be smaller than {top}.')
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


def type_substruct(arg, is_range=False):
    args = arg.split(symbols.COLON)
    type_smiles(args[0])
    match len(args):
        case 1:
            return args[0], None
        case 2:
            if not is_range:
                return args[0], type_float(args[1])
            range_args = args[1].split(symbols.COMMA)
            if len(range_args) == 3:
                return args[0], *[type_float(x) for x in range_args]
            raise argparse.ArgumentTypeError(
                f"{args[1]} should be three numbers separated by comma.")
        case _:
            raise argparse.ArgumentTypeError(
                f"{args} should be smiles and value (or range) separated by colon."
            )


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


def type_force_field(arg, ff_model=symbols.FF_MODEL):
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
    parser.add_argument(FLAG_SEED,
                        metavar=FLAG_SEED[1:].upper(),
                        type=type_random_seed,
                        default=12345,
                        help='Set random state using this seed.')
    parser.add_argument(FLAG_TIMESTEP,
                        metavar='fs',
                        type=type_positive_float,
                        default=1,
                        help=f'Timestep for the MD simulation.')
    parser.add_argument(
        FLAG_STEMP,
        metavar='K',
        type=type_positive_float,
        default=10,
        # 'Initialize the atoms with this temperature.'
        help=argparse.SUPPRESS)
    parser.add_argument(FLAG_TEMP,
                        metavar=FLAG_TEMP[1:].upper(),
                        type=type_nonnegative_float,
                        default=300,
                        help=f'The equilibrium temperature target. A zero '
                        f'value means single point energy calculation.')
    parser.add_argument(
        FLAG_TDAMP,
        metavar=FLAG_TDAMP[1:].upper(),
        type=type_positive_float,
        default=100,
        # Temperature damping parameter (x timestep to get the param)
        help=argparse.SUPPRESS)
    parser.add_argument(FLAG_PRESS,
                        metavar=FLAG_PRESS[1:].upper(),
                        type=float,
                        default=1,
                        help="The equilibrium pressure target.")
    parser.add_argument(
        FLAG_PDAMP,
        metavar=FLAG_PDAMP[1:].upper(),
        type=type_positive_float,
        default=1000,
        # Pressure damping parameter (x timestep to get the param)
        help=argparse.SUPPRESS)
    parser.add_argument(
        FLAG_LJ_CUT,
        metavar=FLAG_LJ_CUT[1:].upper(),
        type=type_positive_float,
        default=lammpsin.In.DEFAULT_LJ_CUT,
        # Cut off for the lennard jones
        help=argparse.SUPPRESS)
    parser.add_argument(
        FLAG_COUL_CUT,
        metavar=FLAG_COUL_CUT[1:].upper(),
        type=type_positive_float,
        default=lammpsin.In.DEFAULT_COUL_CUT,
        # Cut off for the coulombic interaction
        help=argparse.SUPPRESS)
    parser.add_argument(FLAG_RELAX_TIME,
                        metavar='ns',
                        type=type_positive_float,
                        default=1,
                        help='Relaxation simulation time.')
    parser.add_argument(FLAG_PROD_TIME,
                        metavar='ns',
                        type=type_positive_float,
                        default=1,
                        help='Production simulation time.')
    parser.add_argument(FLAG_PROD_ENS,
                        metavar=FLAG_PROD_ENS[1:].upper(),
                        choices=lammpsfix.ENSEMBLES,
                        default=lammpsfix.NVE,
                        help='Production ensemble.')
    parser.add_argument(
        FlAG_FORCE_FIELD,
        metavar=FlAG_FORCE_FIELD[1:].upper(),
        type=type_force_field,
        default=symbols.OPLSUA_TIP3P,
        help='The force field type (and water model separated with comma).')


def add_job_arguments(parser, flags=None, jobname=None):
    """
    Add job control related flags.

    :param parser: the parser to add arguments
    :type parser: 'argparse.ArgumentParser'
    :param flags: specific job control related flags to add
    :type flags: list
    :param jobname: the default jobname
    :type jobname: str
    """
    if flags is None:
        flags = DEFAULT_JOB_FLAGS
    # Workflow drivers may add the job control options a few times
    if FLAG_JOBNAME in flags and FLAG_JOBNAME in parser._option_string_actions:
        parser.set_defaults(jobname=environutils.get_jobname(jobname))
        parser.set_defaults(default_name=jobname)
    flags = [x for x in flags if x not in parser._option_string_actions]
    if FLAG_INTERACTIVE in flags:
        parser.add_argument(FLAG_INTERACTIVE,
                            dest=FLAG_INTERACTIVE[1:].lower(),
                            action='store_true',
                            help='Enable interactive mode')
    if FLAG_JOBNAME in flags:
        parser.add_argument(
            FLAG_JOBNAME,
            dest=FLAG_JOBNAME[1:].lower(),
            default=environutils.get_jobname(jobname),
            help='The jobname based on which filenames are created.')
        parser.add_argument(f"-{jobutils.DEFAULT_NAME}",
                            default=jobname,
                            help=argparse.SUPPRESS)
    if FLAG_DEBUG in flags:
        parser.add_argument(
            FLAG_DEBUG,
            action='store_true',
            dest=FLAG_DEBUG[1:].lower(),
            help='Enable debug mode (e.g. extra printing and files)')
    if FLAG_PYTHON in flags:
        parser.add_argument(
            FLAG_PYTHON,
            default=environutils.CACHE_MODE,
            dest=FLAG_PYTHON[1:].lower(),
            choices=environutils.PYTHON_MODES,
            help='0: pure native python; 1: compile supported python code to '
            'improve performance; 2: run previous compiled python.')
    if FLAG_CPU in flags:
        parser.add_argument(FLAG_CPU,
                            type=type_positive_int,
                            dest=FLAG_CPU[1:].lower(),
                            default=max([round(os.cpu_count() * 0.75), 1]),
                            help='Number of CPU processors.')


def add_workflow_arguments(parser, flags=None):
    """
    Add workflow related flags.

    :param parser: the parser to add arguments
    :type parser: 'argparse.ArgumentParser'
    :param flags: specific workflow related flags to add
    :type flags: list
    """
    if flags is None:
        flags = DEFAULT_WORKFLOW_FLAGS
    if FLAG_STATE_NUM in flags:
        parser.add_argument(
            FLAG_STATE_NUM,
            default=1,
            metavar=FLAG_STATE_NUM[1:].upper(),
            type=type_positive_int,
            help='Number of states for the dynamical system via random seed')
    if FLAG_CLEAN in flags:
        parser.add_argument(
            FLAG_CLEAN,
            action='store_true',
            help='Clean previous workflow results (if any) and run new ones.')
    if FLAG_JTYPE in flags:
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
