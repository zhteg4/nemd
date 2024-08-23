# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module adds jobcontrol related command line flags and job utilities.
"""
import os
import json
import types
from signac import job

from nemd import environutils

RUN_NEMD = 'run_nemd'
ALM = 'alm'
ANPHON = 'anphon'
OUTFILE = 'outfile'
LOGFILE = 'logfile'
OUTFILES = 'outfiles'
FLAG_INTERACTIVE = '-INTERACTIVE'
FLAG_JOBNAME = '-JOBNAME'
DEFAULT_NAME = 'default_name'
FLAG_DEBUG = '-DEBUG'
FLAG_PYTHON = '-PYTHON'
FLAG_SEED = '-seed'
FLAG_CLEAN = '-clean'
FLAG_JTYPE = '-jtype'
FLAG_CPU = '-cpu'
FLAG_PRJ_PATH = '-prj_path'
PREREQ = 'prereq'
FLAG_STATE_NUM = '-state_num'
FLAG_TASK = '-task'
MESSAGE = 'message'

FINISHED = 'Finished.'
FILE = "$FILE"
ARGS = 'args'
TASK = 'task'
AGGREGATOR = 'aggregator'


def get_arg(args, flag, default=None, first=True):
    """
    Get the value after the flag in command arg list.

    :param args: the arg list
    :type args: list
    :param flag: set the value after this flag
    :type flag: str
    :param default: the default if the flag doesn't exist or not followed by value(s)
    :type default: str
    :param first: only return the first value after the flag
    :type first: bool
    :return: the value(s) after the flag
    :rtype: str or list
    """
    try:
        idx = args.index(flag)
    except ValueError:
        # Flag not found
        return default

    val = args[idx + 1]
    if val.startswith('-'):
        # Flag followed by another flag
        return

    if first:
        return val

    selected = []
    for delta, arg in enumerate(args[idx + 1:]):
        if arg.startswith('-'):
            break
        selected.append(arg)
    return selected


def pop_arg(args, flag, val=None):
    """
    Pop the value after the flag in command arg list.

    :param args: the arg list
    :type args: list
    :param flag: set the value after this flag
    :type flag: str
    :param val: the default if the flag doesn't exist or not followed by value(s)
    :type val: str
    :return: the value(s) after the flag
    :rtype: str or list
    """
    arg = get_arg(args, flag)
    if arg is None:
        try:
            args.remove(flag)
        except ValueError:
            pass
        return val

    flag_idx = args.index(flag)
    deta = len(arg) if isinstance(arg, list) else 1
    for idx in reversed(range(flag_idx, flag_idx + deta + 1)):
        args.pop(idx)
    return arg


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
                document=job.Job.FN_DOCUMENT,
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


def get_sidx(data, options, delta=1):
    """
    Get the start index from last_pct.

    :param data: the input data on which the length is determined
    :type data: list, tuple, or numpy.ndarray
    :param options: the percentage of the last frames
    :type options: 'argparse.ArgumentParser'
    :param delta: the minimum number of frames to include in the selection
    :type delta: int
    :return: the start index
    :rtype: int
    """
    num = len(data)
    sidx = round(num * (1 - options.last_pct))
    max_sidx = num - delta if num - delta > 0 else 0
    return min(sidx, max_sidx)


class Job:
    """
    A class to mimic a signac.job.Job.
    """

    def __init__(self, job_dir=os.curdir):
        """
        Initialize a Job object.

        :param job_dir: the directory of the job
        :type job_dir: str
        """
        self.dir = job_dir
        self.statepoint = {}
        self.doc = {}
        self.document = self.doc
        self.project = types.SimpleNamespace(doc={},
                                             workspace='workspace',
                                             path=os.curdir)
        doc = os.path.join(self.dir, job.Job.FN_DOCUMENT)
        if os.path.isfile(doc):
            with open(doc, 'r') as dfh:
                self.doc.update(json.load(dfh))
        statepoint = os.path.join(self.dir, job.Job.FN_STATE_POINT)
        if os.path.isfile(statepoint):
            with open(statepoint, 'r') as sfh:
                self.statepoint.update(json.load(sfh))

    def fn(self, filename):
        """
        Return the full path of the file in the job directory.

        :param filename str: the file name
        :return str: the full path of the file
        """
        return os.path.join(self.dir, filename) if self.dir else filename
