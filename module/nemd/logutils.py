import os
import logging
import pathlib
from datetime import datetime

from nemd import environutils

def createLogger(basename, verbose=None, file_ext='-driver.log'):
    if verbose is None:
        verbose = environutils.is_debug()
    logger = logging.getLogger(basename)
    log_filename = basename + file_ext
    if os.path.isfile(log_filename):
        os.remove(log_filename)
    hdlr = logging.FileHandler(log_filename)
    logger.addHandler(hdlr)

    if verbose:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    else:
        formatter = logging.Formatter('%(message)s')

    hdlr.setFormatter(formatter)
    lvl = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(lvl)
    return logger


def createDriverLogger(jobname, verbose=None):
    return createLogger(jobname, verbose=verbose)


def createModuleLogger(basename=None,
                       verbose=True,
                       file_path=None,
                       file_ext='.log'):
    if not environutils.is_debug():
        return

    if basename:
        return createLogger(basename, verbose=verbose, file_ext=file_ext)

    if not file_path:
        raise ValueError(f"Either basename or file_path should be provided.")

    file_path = pathlib.Path(file_path)
    module_path = environutils.get_module_path()
    if module_path:
        relpath = os.path.relpath(file_path, environutils.get_nemd_src())
        basename = relpath.replace(os.path.sep, '.')
    else:
        basename = str(file_path.name)
    if basename.endswith('.py'):
        basename = basename[:-3]
    return createLogger(basename, verbose=verbose, file_ext=file_ext)


def logOptions(logger, options):
    command_options = 'Command Options'
    logger.info(f"." * 10 + command_options + f"." * 10)
    for key, val in options.__dict__.items():
        logger.info(f"{key}: {val}")
    time = datetime.now().isoformat(sep=' ', timespec='minutes')
    logger.info(f"JobStart: {time}")
    logger.info(f"." * (20 + len(command_options)))


def log(logger, msg, timestamp=False):
    logger.info(msg)
    if timestamp:
        time = datetime.now().isoformat(sep=' ', timespec='minutes')
        logger.info(time)
