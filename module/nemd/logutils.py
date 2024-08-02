import os
import sh
import io
import logging
import pathlib
import wurlitzer
import contextlib

from nemd import environutils
from nemd import timeutils
from nemd import jobutils

DRIVER_LOG = '-driver.log'
JOBSTART = 'JobStart:'
FINISHED = 'Finished.'
START = 'start'
END = 'end'
DELTA = 'delta'


class FileHandler(logging.FileHandler):
    """
    Handler that controls the writing of the newline character.

    https://stackoverflow.com/questions/7168790/suppress-newline-in-python-logging-module
    """

    NO_NEWLINE = '[!n]'

    def emit(self, record):
        """
        See parent method for documentation.
        """
        newline = not record.msg.endswith(self.NO_NEWLINE)
        pre_newline = self.terminator == '\n'
        self.terminator = '\n' if newline else ''
        record.msg = record.msg.replace(self.NO_NEWLINE, '')
        if not pre_newline:
            record.msg = self.NO_NEWLINE + record.msg
        return super().emit(record)

    def format(self, record):
        """
        See parent method for documentation.
        """
        default = not self.formatter or record.msg.startswith(self.NO_NEWLINE)
        fmt = logging._defaultFormatter if default else self.formatter
        record.msg = record.msg.replace(self.NO_NEWLINE, '')
        return fmt.format(record)


def createLogger(basename, verbose=None, file_ext=DRIVER_LOG, log_file=False):
    """
    Create a logger.

    :param basename: the basename of the job
    :type basename: str
    :param verbose: extra info printed out (e.g. debug level info) if True
    :type verbose: bool
    :param file_ext: the extension of the logger file
    :type file_ext: str
    :return: the logger
    :param log_file: sets as the log file if True
    :type log_file: bool
    :rtype: 'logging.Logger'
    """
    if verbose is None:
        verbose = environutils.is_debug()

    logger = logging.getLogger(basename)
    log_filename = basename + file_ext
    if os.path.isfile(log_filename):
        try:
            os.remove(log_filename)
        except FileNotFoundError:
            pass
    if log_file:
        jobutils.add_outfile(log_filename, jobname=basename, log_file=log_file)
    hdlr = FileHandler(log_filename)
    fmt = '%(asctime)s %(levelname)s %(message)s' if verbose else '%(message)s'
    hdlr.setFormatter(logging.Formatter(fmt))
    logger.addHandler(hdlr)
    lvl = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(lvl)
    return logger


def createDriverLogger(jobname, verbose=None, log_file=True):
    """
    Create a driver logger.

    :param jobname: the jobname
    :type jobname: str
    :param verbose: extra info printed out (e.g. debug level info) if True
    :type verbose: bool
    :param log_file: sets as the log file if True
    :type log_file: bool
    :return: the logger
    :rtype: 'logging.Logger'
    """
    return createLogger(jobname, verbose=verbose, log_file=log_file)


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
        if type(val) is list:
            val = ', '.join(map(str, val))
        logger.info(f"{key}: {val}")
    logger.info(f"{JOBSTART} {timeutils.ctime()}")
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
        logger.info(timeutils.ctime())


def get_time(filepath, dtype=DELTA):
    """
    Get the time information from log file.

    :param filepath: the log filepath
    :type filepath: str
    :param dtype: START gets the starting time, END get the finishing time,
        and DELTA gets the time span.
    :type dtype: str
    :return: the time information
    :rtype: 'datetime.datetime' on START & END; 'datetime.timedelta' on DELTA
    """
    stime = sh.grep(JOBSTART, filepath).split(JOBSTART)[-1].strip()
    stime = timeutils.dtime(stime)
    if dtype == START:
        return stime
    try:
        dtime = sh.grep('-A', '1', FINISHED, filepath).split(FINISHED)[-1]
    except sh.ErrorReturnCode_1:
        return
    dtime = timeutils.dtime(' '.join(dtime.split()[-2:]))
    if dtype == END:
        return dtime
    delta = dtime - stime
    return delta


@contextlib.contextmanager
def redirect(*args, logger=None, **kwds):
    """
    Redirecting all kinds of stdout in Python via wurlitzer
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    :param logger 'logging.Logger': the logger to print the out and err messages.
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with wurlitzer.pipes(out, err) as handler:
            yield handler
    finally:
        if logger is None:
            return
        out = out.getvalue()
        if out:
            logger.warning(out)
        err = err.getvalue()
        if err:
            logger.warning(err)
