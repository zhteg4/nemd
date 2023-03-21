import os
import pkgutil

NEMD_DEBUG = 'NEMD_DEBUG'
NEMD_SRC = 'NEMD_SRC'
DEBUG = 'DEBUG'
JOBNAME = 'JOBNAME'
INTERACTIVE = 'INTERACTIVE'
NEMD = 'nemd'
TEST = 'test'
MODULE = 'module'
INTEGRATION = 'integration'


def is_debug():
    """
    Whether the execution is in the debug mode.

    :return bool: If the debug environment flag is on.

    NOTE: NEMD_DEBUG is designed as command environment flag while DEBUG is
    usually fed into specific execution command.
    """
    return bool(os.environ.get(NEMD_DEBUG)) or os.environ.get(DEBUG)


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