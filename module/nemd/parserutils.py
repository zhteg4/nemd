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


class CapitalisedHelpFormatter(argparse.HelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '
        return super(CapitalisedHelpFormatter,
                     self).add_usage(usage, actions, groups, prefix)


def get_parser(**kwargs):
    return argparse.ArgumentParser(formatter_class=CapitalisedHelpFormatter,
                                   **kwargs)


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
    raise argparse.ArgumentTypeError(f"None of {', '.join([arg] + nargs)} exists.")


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
