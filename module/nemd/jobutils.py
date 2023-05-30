# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module adds jobcontrol related command line flags and job utilities.
"""
import json
from signac.contrib import job
from nemd import environutils

RUN_NEMD = 'run_nemd'
OUTFILE = 'outfile'
LOGFILE = 'logfile'
OUTFILES = 'outfiles'
TARGS = 'targs'
FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
FLAG_DEBUG = '-DEBUG'
FLAG_SEED = '-seed'
FLAG_CLEAN = '-clean'
FLAG_JTYPE = '-jtype'
FLAG_CPU = '-cpu'
FLAG_PRJ_PATH = '-prj_path'
PREREQ = 'prereq'

FINISHED = 'Finished.'
FILE = "$FILE"
ARGS = 'args'
KNOWN_ARGS = 'known_args'
UNKNOWN_ARGS = 'unknown_args'
FN_DOCUMENT = job.Job.FN_DOCUMENT
TASK = 'task'
AGGREGATOR = 'aggregator'


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
                job=None,
                document=FN_DOCUMENT,
                set_file=False,
                log_file=False):
    """
    Register the outfile to the job control.

    :param outfile: the outfile to be registered
    :type outfile: str
    :param jobname: register the file under this jobname
    :type jobname: str
    :param default_jobname: use this jobname if jobname is undefined and cannot
        be found from the environment
    :type default_jobname: str
    :param job: register outfile to this job document
    :type job: 'signac.contrib.job.Job'
    :param document: the job control information is saved into this file if job
        not provided
    :type document: str
    :param set_file: set this file as the single output file
    :type set_file: bool
    :param log_file: set this file as the log file
    :type log_file: bool
    """
    if jobname is None:
        jobname = environutils.get_jobname(default_jobname)
    if job:
        data = job.document
    else:
        try:
            with open(document) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            data = {}
    data.setdefault(OUTFILES, {})
    data[OUTFILES].setdefault(jobname, [])
    if outfile not in data[OUTFILES][jobname]:
        data[OUTFILES][jobname].append(outfile)
    if set_file:
        data.setdefault(OUTFILE, {})
        data[OUTFILE].setdefault(jobname, outfile)
    if log_file:
        data.setdefault(LOGFILE, {})
        data[LOGFILE].setdefault(jobname, outfile)
    if job:
        return
    with open(document, 'w') as fh:
        json.dump(data, fh)
