import os
import logging
import pathlib
from datetime import datetime

from nemd import environutils

DRIVER_LOG = '-driver.log'


def createLogger(basename, verbose=None, file_ext=DRIVER_LOG):
    """
    Create a logger.

    :param basename: the basename of the job
    :type basename: str
    :param verbose: extra info printed out (e.g. debug level info) if True
    :type verbose: bool
    :param file_ext: the extension of the logger file
    :type file_ext: str
    :return: the logger
    :rtype: 'logging.Logger'
    """
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
    """
    Create a driver logger.

    :param jobname: the jobname
    :type jobname: str
    :param verbose: extra info printed out (e.g. debug level info) if True
    :type verbose: bool
    :return: the logger
    :rtype: 'logging.Logger'
    """
    return createLogger(jobname, verbose=verbose)


def createModuleLogger(basename=None,
                       verbose=True,
                       file_path=None,
                       file_ext='.log'):
    """
    Create logger for module so that debug printing can be enabled.

    :param basename: the basename of the job
    :type basename: str
    :param verbose: extra info printed out (e.g. debug level info) if True
    :type verbose: bool
    :param file_path: module file path based on which logger name is obtained
    :type file_path: str
    :param file_ext: the extension of the logger file
    :type file_ext: str
    """
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
    """
    Print description of the options.

    :param logger:  print to this logger
    :type logger: 'logging.Logger'
    :param options: command-line options
    :type options: 'argparse.Namespace'
    """
    command_options = 'Command Options'
    logger.info(f"." * 10 + command_options + f"." * 10)
    for key, val in options.__dict__.items():
        logger.info(f"{key}: {val}")
    time = datetime.now().isoformat(sep=' ', timespec='minutes')
    logger.info(f"JobStart: {time}")
    logger.info(f"." * (20 + len(command_options)))


def log(logger, msg, timestamp=False):
    """
    Log message to the logger.

    :param logger:  print to this logger
    :type logger: 'logging.Logger'
    :param msg: the message to be printed out
    :type msg: str
    :param timestamp: append time information after the message
    :type timestamp: bool
    """
    logger.info(msg)
    if timestamp:
        time = datetime.now().isoformat(sep=' ', timespec='minutes')
        logger.info(time)
