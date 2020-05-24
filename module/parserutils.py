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
