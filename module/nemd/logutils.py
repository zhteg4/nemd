import os
import io
import types
import logging
import pathlib
import wurlitzer
import contextlib

from nemd import symbols
from nemd import environutils
from nemd import timeutils
from nemd import jobutils

DRIVER_LOG = '-driver.log'
JOBSTART = 'JobStart:'
FINISHED = 'Finished.'
START = 'start'
END = 'end'
DELTA = 'delta'
COMMAND_OPTIONS = 'Command Options'
COMMAND_OPTIONS_START = f"." * 10 + COMMAND_OPTIONS + f"." * 10
COMMAND_OPTIONS_END = f"." * (20 + len(COMMAND_OPTIONS))
COLON_SEP = f'{symbols.COLON} '
COMMA_SEP = symbols.COMMA_SEP


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


def createLogger(basename,
                 verbose=None,
                 file_ext=DRIVER_LOG,
                 log_file=False,
                 set_file=False):
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
    :param set_file: set this file as the single output file
    :type set_file: bool
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
        jobutils.add_outfile(log_filename,
                             jobname=basename,
                             log_file=log_file,
                             set_file=set_file)
    hdlr = FileHandler(log_filename)
    fmt = '%(asctime)s %(levelname)s %(message)s' if verbose else '%(message)s'
    hdlr.setFormatter(logging.Formatter(fmt))
    logger.addHandler(hdlr)
    lvl = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(lvl)
    return logger


def createDriverLogger(jobname, verbose=None, log_file=True, set_file=False):
    """
    Create a driver logger.

    :param jobname: the jobname
    :type jobname: str
    :param verbose: extra info printed out (e.g. debug level info) if True
    :type verbose: bool
    :param log_file: sets as the log file if True
    :type log_file: bool
    :param set_file: set this file as the single output file
    :type set_file: bool
    :return: the logger
    :rtype: 'logging.Logger'
    """
    return createLogger(jobname,
                        verbose=verbose,
                        log_file=log_file,
                        set_file=set_file)


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
    logger.info(COMMAND_OPTIONS_START)
    for key, val in options.__dict__.items():
        if type(val) is list:
            val = COMMA_SEP.join(map(str, val))
        logger.info(f"{key}{COLON_SEP}{val}")
    logger.info(f"{JOBSTART} {timeutils.ctime()}")
    logger.info(COMMAND_OPTIONS_END)


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


@contextlib.contextmanager
def redirect(*args, logger=None, **kwargs):
    """
    Redirecting all kinds of stdout in Python via wurlitzer
    https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

    :param logger 'logging.Logger': the logger to print the out and err messages.
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with wurlitzer.pipes(out, err):
            yield None
    finally:
        if logger is None:
            return
        out = out.getvalue()
        if out:
            logger.warning(out)
        err = err.getvalue()
        if err:
            logger.warning(err)


class LogReader:
    """
    A class to read the log file.
    """

    INFO_SEP = ' INFO '
    TOTOAL_TIME = 'Task Total Timing: '

    def __init__(self, filepath, delay=False):
        """
        Initialize the LogReader object.

        :param filepath: the log filepath
        :type filepath: str
        :param delay: if True, delay the reading of the log file
        :type delay: bool
        """
        self.filepath = filepath
        self.lines = None
        self.options = None
        self.sidx = None
        self.delay = delay
        if self.delay:
            return
        self.run()

    def run(self):
        """
        Run the log reader.
        """
        self.read()
        self.setOptions()

    def read(self):
        """
        Read the log file.
        """
        with open(self.filepath, 'r') as fh:
            self.lines = [
                x.split(self.INFO_SEP)[-1].strip() for x in fh.readlines()
            ]

    def setOptions(self):
        """
        Set the options from the log file.
        """
        block = None
        for idx, line in enumerate(self.lines):
            if line.endswith(COMMAND_OPTIONS_END):
                self.sidx = idx + 1
                break
            if block is not None:
                block.append(line)
            if line.endswith(COMMAND_OPTIONS_START):
                block = []
        options = {}
        for line in block:
            key, val = line.split(COLON_SEP)
            key = key.split()[-1]
            vals = val.split(COMMA_SEP)
            options[key] = val if len(vals) == 1 else vals
        self.options = types.SimpleNamespace(**options)

    @property
    def task_time(self):
        """
        Get the task time from log file.

        :return: the task time
        :rtype: 'datetime.timedelta'
        """
        for line in self.lines[self.sidx:]:
            if not line.startswith(self.TOTOAL_TIME):
                continue
            task_time = line.split(self.TOTOAL_TIME)[-1].strip()
            return timeutils.str2delta(task_time)
        return self.time()

    def time(self, dtype=DELTA):
        """
        Get the time information from log file.

        :param dtype: START gets the starting time, END get the finishing time,
            and DELTA gets the time span.
        :type dtype: str
        :return: the time information
        :rtype: 'datetime.datetime' on START & END; 'datetime.timedelta' on DELTA
        """
        stime = timeutils.dtime(self.options.JobStart)
        if dtype == START:
            return stime
        # len('12:54:16 08/15/24') == 17
        dtime = timeutils.dtime(self.lines[-1][-17:])
        if dtype == END:
            return dtime
        delta = dtime - stime
        return delta


class Base(object):
    """
    A base class with a logger to print logging messages.
    """

    def __init__(self, logger=None):
        """
        :param logger: the logger to log messages
        :type logger: 'logging.Logger'
        """
        self.logger = logger

    def log(self, msg, **kwargs):
        """
        Print this message into the log file as information.

        :param msg str: the msg to be printed
        """
        if self.logger:
            log(self.logger, msg, **kwargs)
        else:
            print(msg)

    def log_debug(self, msg):
        """
        Print this message into the log file in debug mode.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.debug(msg)
        else:
            print(msg)

    def log_warning(self, msg):
        """
        Print this warning message into log file.

        :param msg str: the msg to be printed
        """
        if self.logger:
            self.logger.warning(msg)
        else:
            print(msg)
