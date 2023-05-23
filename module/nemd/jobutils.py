# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module adds jobcontrol related command line flags and job utilities.
"""
import json
from nemd import environutils
from signac.contrib import job

RUN_NEMD = 'run_nemd'
OUTFILE = 'outfile'
OUTFILES = 'outfiles'
FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
FLAG_DEBUG = '-DEBUG'
FLAG_SEED = '-seed'
FLAG_CLEAN = '-clean'
PREREQ = 'prereq'

FINISHED = 'Finished.'
ARGS = 'args'
KNOWN_ARGS = 'known_args'
UNKNOWN_ARGS = 'unknown_args'


def add_job_arguments(parser, arg_flags=None):
    """
    Add job control related flags.

    :param parser: the parser to add arguments
    :type parser: 'argparse.ArgumentParser'
    :param arg_flags: specific job control related flags to add
    :type arg_flags: list
    """
    if arg_flags is None:
        arg_flags = [FLAG_INTERACTIVE, FLAG_JOBNAME, FLAG_DEBUG]
    # Workflow drivers may add the job control options a few times
    arg_flags = [
        x for x in arg_flags if x not in parser._option_string_actions
    ]
    if FLAG_INTERACTIVE in arg_flags:
        parser.add_argument(FLAG_INTERACTIVE,
                            dest=FLAG_INTERACTIVE[1:].lower(),
                            action='store_true',
                            help='Enable interactive mode')
    if FLAG_JOBNAME in arg_flags:
        parser.add_argument(FLAG_JOBNAME,
                            dest=FLAG_JOBNAME[1:].lower(),
                            help='The jobnamee based on which filenames are created.')
    if FLAG_DEBUG in arg_flags:
        parser.add_argument(FLAG_DEBUG,
                            action='store_true',
                            dest=FLAG_DEBUG[1:].lower(),
                            help='Enable debug mode (e.g. extra printing and files)')
    if FLAG_CLEAN in arg_flags:
        parser.add_argument(
            FLAG_CLEAN,
            action='store_true',
            help='Clean previous workflow results (if any) and run new ones.')


def get_arg(args, flag, val=None):
    """
    Get the value after the flag in command arg list.

    :param args: the arg list
    :type args: list
    :param flag: set the value after this flag
    :type flag: str
    :param val: the default value if the flag doesn't exist
    :type val: str
    :return: the value after the flag
    :rtype: str
    """
    try:
        idx = args.index(flag)
    except ValueError:
        return val
    else:
        return args[idx + 1]


def set_arg(args, flag, val):
    """
    Set the value after the flag in command arg list.

    :param args: the arg list
    :type args: list
    :param flag: set the value after this flag
    :type flag: str
    :param val: the new value
    :type val: str
    :return: the modified arg list
    :rtype: list
    """
    if flag not in args:
        args.extend([flag, ''])
    idx = args.index(flag)
    args[idx + 1] = val
    return args


def add_outfile(outfile,
                jobname=None,
                default_jobname=None,
                document=job.Job.FN_DOCUMENT,
                set_file=False):
    """
    Register the outfile to the job control.

    :param outfile: the outfile to be registered
    :type outfile: str
    :param jobname: register the file under this jobname
    :type jobname: str
    :param default_jobname: use this jobname if jobname is undefined and cannot
        be found from the environment
    :type default_jobname: str
    :param document: the job control information is saved in this file
    :type document: str
    :param set_file: set this file as the single output file
    :type set_file: bool
    """
    if jobname is None:
        jobname = environutils.get_jobname(default_jobname)
    try:
        with open(document) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = {}
    data.setdefault(OUTFILES, {})
    data[OUTFILES].setdefault(jobname, [])
    data[OUTFILES][jobname].append(outfile)
    if set_file:
        data.setdefault(OUTFILE, {})
        data[OUTFILE].setdefault(jobname, outfile)
    with open(document, 'w') as fh:
        json.dump(data, fh)
