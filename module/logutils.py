import os
import logging
import pathlib


NEMD_DEBUG = 'NEMD_DEBUG'

def in_debug():
    return bool(os.environ.get(NEMD_DEBUG))

def createLogger(basename, verbose=None, file_ext='-driver.log'):
    if verbose is None:
        verbose = in_debug()
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
    logger.setLevel(logging.DEBUG)
    return logger

def createDriverLogger(jobname, verbose=None):
    return createLogger(jobname, verbose=verbose)

def createModuleLogger(basename=None, verbose=True, file_ext='.log'):
    if not in_debug():
        return

    if basename:
        return createLogger(basename, verbose=verbose, file_ext=file_ext)

    file_path = pathlib.Path(__file__)
    nemd_src = os.environ.get('NEMD_SRC')
    if nemd_src:
        module_path = os.path.join(nemd_src, 'module')
        file_relpath = str(file_path.relative_to(module_path))
        basename = file_relpath.replace(os.path.sep, '.')
    else:
        basename = str(file_path.name)
    if basename.endswith('.py'):
        basename = basename[:-3]
    else:
        basename += file_ext

    return createLogger(basename, verbose=verbose, file_ext=file_ext)

def logOptions(logger, options):
    logger.info(f"Command Options:")
    logger.info(f"")
    for key, val in options.__dict__.items():
        logger.info(f"{key}: {val}")
    logger.info(f"")