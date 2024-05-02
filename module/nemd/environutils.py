# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module handles system-wide variables from the shell environment.
"""
import os
import pkgutil

NEMD_DEBUG = 'NEMD_DEBUG'
NEMD_SRC = 'NEMD_SRC'
DEBUG = 'DEBUG'
PYTHON = 'PYTHON'
PYTHON_MODE = -1
ORIGINAL_MODE = 0
NOPYTHON_MODE = 1
CACHE_MODE = 2
PYTHON_MODES = [PYTHON_MODE, ORIGINAL_MODE, NOPYTHON_MODE, CACHE_MODE]
JOBNAME = 'JOBNAME'
INTERACTIVE = 'INTERACTIVE'
NEMD = 'nemd'
TEST = 'test'
MODULE = 'module'
SUBMODULE = 'submodule'
INTEGRATION = 'integration'

ALAMODE = 'alamode'


def is_debug():
    """
    Whether the execution is in the debug mode.

    :return bool: If the debug environment flag is on.

    NOTE: NEMD_DEBUG is designed as command environment flag while DEBUG is
    usually fed into specific execution command.
    """
    return bool(os.environ.get(NEMD_DEBUG)) or os.environ.get(DEBUG)


def get_python_mode():
    """
    Get the mode of python compilation.

    :return int: The mode of python compilation as follows:
        0: pure native python;
        1:compile supported python code to improve performance;
        2: cache compiled python code.
    """
    return int(os.environ.get(PYTHON, CACHE_MODE))


def get_nemd_src():
    """
    Get the source code dir.

    :return str: the source code dir
    """
    return os.environ.get('NEMD_SRC')


def get_module_path():
    """
    Get the module path.

    NOTE: If installed, all modules are assumed to sit together. In dev mode,
    the module is search

    :return str: the module path
    """
    nemd_src = get_nemd_src()
    if not nemd_src:
        return os.path.dirname(pkgutil.get_loader(NEMD).path)
    return os.path.join(nemd_src, MODULE, NEMD)


def get_submodule_path():
    """
    Get the module path.

    NOTE: If installed, all modules are assumed to sit together. In dev mode,
    the module is search

    :return str: the module path
    """
    nemd_src = get_nemd_src()
    if not nemd_src:
        return os.path.dirname(pkgutil.get_loader(NEMD).path)
    return os.path.join(nemd_src, SUBMODULE)


def get_integration_test_dir():
    """
    Get the module path.

    NOTE: If installed, all modules are assumed to sit together. In dev mode,
    the module is search

    :return str: the module path
    """
    nemd_src = get_nemd_src()
    if not nemd_src:
        raise ValueError(f'Please set {NEMD_SRC} pointing to the source code.')
    return os.path.join(nemd_src, TEST, INTEGRATION)


def get_jobname(default_jobname):
    """
    Get the joabname from environment settings or the input default one.

    :param default_jobname str: the default one if the environment one is missing.

    :return str: the jobname.
    """
    jobname = os.environ.get(JOBNAME)
    if jobname:
        return jobname
    return default_jobname


def is_interactive():
    """
    Whether interactive mode is on.

    :return bool: If interactive mode is on.
    """
    return os.environ.get(INTERACTIVE)
