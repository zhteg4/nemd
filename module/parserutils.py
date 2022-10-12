import os
import argparse


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


def type_positive_float(arg):
    try:
        value = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{arg} cannot be converted to float.')
    if value <= 0:
        raise argparse.ArgumentTypeError(f'{value} is not a possitive float.')
    return value


def type_positive_int(arg):
    try:
        value = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'{arg} cannot be converted to integer.')
    if value <= 1:
        raise argparse.ArgumentTypeError(
            f'{value} is not a possitive integer.')
    return value
