import os

NEMD_DEBUG = 'NEMD_DEBUG'
NEMD_SRC = 'NEMD_SRC'


def is_debug():
    return bool(os.environ.get(NEMD_DEBUG))


def get_nemd_src():
    return os.environ.get('NEMD_SRC')


def get_module_path():
    nemd_src = get_nemd_src()
    if not nemd_src:
        return
    return os.path.join(nemd_src, 'module')

def get_jobname(default_jobname):
    jobname = os.environ.get('JOBNAME')
    if jobname:
        return jobname
    return default_jobname

def is_interactive():
    return os.environ.get('INTERACTIVE')