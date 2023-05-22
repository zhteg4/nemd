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
FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
FLAG_DEBUG = '-DEBUG'
FLAG_SEED = '-seed'
FINISHED = 'Finished.'
ARGS = 'args'
KNOWN_ARGS = 'known_args'
UNKNOWN_ARGS = 'unknown_args'


def add_job_arguments(parser, arg_flags=None):
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


def get_arg(args, flag, val=None):
    try:
        idx = args.index(flag)
    except ValueError:
        return val
    else:
        return args[idx + 1]


def set_arg(args, flag, val):
    if flag not in args:
        args.extend([flag, ''])
    idx = args.index(flag)
    args[idx + 1] = val
    return args


def add_outfile(outfile,
                jobname=None,
                default_jobname=None,
                document=job.Job.FN_DOCUMENT):
    if jobname is None:
        jobname = environutils.get_jobname(default_jobname)
    try:
        with open(document) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        data = {}
    data.setdefault(OUTFILE, {})
    data[OUTFILE].setdefault(jobname, [])
    data[OUTFILE][jobname].append(outfile)
    with open(document, 'w') as fh:
        json.dump(data, fh)
