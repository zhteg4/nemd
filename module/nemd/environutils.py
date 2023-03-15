import os
import pkgutil

NEMD_DEBUG = 'NEMD_DEBUG'
NEMD_SRC = 'NEMD_SRC'
DEBUG = 'DEBUG'


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
        return os.path.dirname(pkgutil.get_loader('nemd').path)
    return os.path.join(nemd_src, 'module', 'nemd')


def get_jobname(default_jobname):
    jobname = os.environ.get('JOBNAME')
    if jobname:
        return jobname
    return default_jobname


def is_interactive():
    return os.environ.get('INTERACTIVE')
