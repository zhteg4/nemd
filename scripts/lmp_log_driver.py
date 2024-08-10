# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This post molecular dynamics driver perform log analysis.
"""
import os
import sys
import math
import functools

from nemd import symbols
from nemd import stillinger
from nemd import lammpslog
from nemd import jobutils
from nemd import logutils
from nemd import analyzer
from nemd import parserutils
from nemd import environutils

FlAG_LMP_LOG = 'lmp_log'
FLAG_DATA_FILE = parserutils.FLAG_DATA_FILE
FLAG_TASK = jobutils.FLAG_TASK
FLAG_LAST_PCT = '-last_pct'
FLAG_SLICE = '-slice'

ALL_FRM_TASKS = analyzer.Thermo.TASKS
LAST_FRM_TASKS = ALL_FRM_TASKS
AVE_FRM_TASKS = LAST_FRM_TASKS
DATA_RQD_TASKS = []
NO_COMBINE = []

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')
# Positional command-argument holders to take task input under jobcontrol
ARGS_TMPL = [jobutils.FILE]


def log_debug(msg):
    """
    Print this message into the log file in debug mode.

    :param msg str: the msg to be printed
    """
    if logger:
        logger.debug(msg)


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: print time after the msg
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


def log_warning(msg):
    """
    Print this warning message into log file.

    :param msg: the msg to print
    """
    if not logger:
        return
    logger.warning(msg)


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


class LmpLog(object):
    """
    Analyze a lammps log.
    """

    TASK = FLAG_TASK[1:]
    DATA_EXT = '_%s.csv'
    AVE_DATA_EXT = '_ave' + DATA_EXT
    PNG_EXT = '_%s.png'
    RESULTS = analyzer.Thermo.RESULTS

    def __init__(self, options):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.options = options
        self.lmp_log = None
        self.df_reader = None
        self.tasks = [
            x for x in analyzer.Thermo.TASKS if x in self.options.task
        ]

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setThermo()
        self.setTasks()
        self.analyze()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return

        self.df_reader = stillinger.DataFileReader.from_file(
            self.options.data_file)

    def setThermo(self):
        """
        Grep thermo output information.
        """

        self.lmp_log = lammpslog.Log(self.options.lmp_log)
        self.lmp_log.run()
        self.sidx = math.floor(self.lmp_log.thermo.shape[0] *
                               (1 - self.options.last_pct))
        self.lmp_log.thermo.index.name += f" ({self.sidx})"

        log(f"{self.lmp_log.thermo.shape[0]} steps of thermo data found.")
        af_tasks = [x for x in self.options.task if x in ALL_FRM_TASKS]
        if af_tasks:
            log(f"{', '.join(af_tasks)} collects and saves results "
                f"{symbols.ELEMENT_OF} [{self.lmp_log.thermo.index[0]:.3f}, "
                f"{self.lmp_log.thermo.index[0]:.3f}] ps")
        lf_tasks = [x for x in self.options.task if x in LAST_FRM_TASKS]
        if lf_tasks:
            log(f"{', '.join(lf_tasks)} averages results from last "
                f"{self.options.last_pct * 100}% frames {symbols.ELEMENT_OF} "
                f"[{self.lmp_log.thermo.index[self.sidx]:.3f}, "
                f"{self.lmp_log.thermo.index[-1]:.3f}] ps")

    def setTasks(self):
        """
        Set the tasks to be performed.
        """
        columns = self.lmp_log.thermo.columns
        available = [analyzer.Base.parseLabel(x)[0].lower() for x in columns]
        selected = set(self.tasks).intersection(available)
        if len(selected) == len(self.tasks):
            return
        missed = symbols.COMMA_SEP.join(set(self.tasks).difference(selected))
        available = symbols.COMMA_SEP.join(available)
        log_warning(f"{missed} tasks cannot be found out of {available}.")
        self.tasks = list(selected)

    def analyze(self):
        """
        Run analyzers.
        """
        for task in self.tasks:
            anl = analyzer.ANALYZER[task](self.lmp_log.thermo,
                                          options=self.options,
                                          logger=logger,
                                          df_reader=self.df_reader)
            anl.run()


def get_parser(parser=None, jflags=None):
    """
    The user-friendly command-line parser.

    :param parser ArgumentParser: the parse to add arguments
    :param jflags list: specific job control related flags to add
    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
        parser.add_argument(FlAG_LMP_LOG,
                            metavar=FlAG_LMP_LOG.upper(),
                            type=parserutils.type_file,
                            help='LAMMPS log file to analyze')
        parser.add_argument(FLAG_DATA_FILE,
                            metavar=FLAG_DATA_FILE[1:].upper(),
                            type=parserutils.type_file,
                            help='Data file to get force field information')
    parser.add_argument(FLAG_TASK,
                        choices=analyzer.Thermo.TASKS,
                        default=analyzer.Thermo.TOTENG,
                        nargs='+',
                        help=f'{analyzer.Thermo.NAME} searches, combines and '
                        f'averages thermodynamic info. ')
    parser.add_argument(
        FLAG_LAST_PCT,
        metavar=FLAG_LAST_PCT.upper(),
        type=functools.partial(parserutils.type_ranged_float,
                               include_top=False,
                               top=1),
        default=0.2,
        help=f"{', '.join(LAST_FRM_TASKS)} average results from "
        f"last thermo output of this percentage.")
    parser.add_argument(FLAG_SLICE,
                        metavar='START:END:INTERVAL',
                        type=parserutils.type_slice,
                        help=f"Slice the thermo output for analysis.")
    parserutils.add_job_arguments(parser,
                                  arg_flags=jflags,
                                  jobname=environutils.get_jobname(JOBNAME))
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    data_rqd_tasks = set(options.task).intersection(DATA_RQD_TASKS)
    if data_rqd_tasks and not options.data_file:
        parser.error(f"Please specify {FLAG_DATA_FILE} to run {FLAG_TASK} "
                     f"{', '.join(data_rqd_tasks)}")
    return options


logger = None


def main(argv):

    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname,
                                         log_file=True)
    logutils.logOptions(logger, options)
    lmp_log = LmpLog(options)
    lmp_log.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
