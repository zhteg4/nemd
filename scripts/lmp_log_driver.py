# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This post molecular dynamics driver perform log analysis.
"""
import re
import sh
import os
import sys
import functools

from nemd import traj
from nemd import symbols
from nemd import stillinger
from nemd import fileutils
from nemd import jobutils
from nemd import logutils
from nemd import analyzer
from nemd import parserutils
from nemd import environutils

FlAG_LMP_LOG = 'lmp_log'
FlAG_DATA_FILE = traj.FlAG_DATA_FILE
FlAG_TASK = '-task'
FLAG_LAST_PCT = '-last_pct'
FLAG_SLICE = '-slice'

THERMO = 'thermo'
TEMP = 'Temp'
EPAIR = 'E_pair'
E_MOL = 'E_mol'
TOTENG = 'TotEng'
PRESS = 'Press'
THERMO_TASKS = [TEMP, EPAIR, EPAIR, TOTENG, PRESS]

ALL_FRM_TASKS = THERMO_TASKS
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


def log_error(msg):
    """
    Print this message and exit the program.

    :param msg str: the msg to be printed
    """
    log(msg + '\nAborting...', timestamp=True)
    sys.exit(1)


class LmpLog(object):
    """
    Analyze a dump custom file.
    """

    TASK = FlAG_TASK[1:]
    DATA_EXT = '_%s.csv'
    PNG_EXT = '_%s.png'
    RESULTS = analyzer.BaseAnalyzer.RESULTS
    ANALYZER = [
        analyzer.Density, analyzer.RDF, analyzer.MSD, analyzer.Clash,
        analyzer.View, analyzer.XYZ
    ]
    ANALYZER = {getattr(x, 'NAME'): x for x in ANALYZER}

    def __init__(self, options):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        """
        self.options = options
        self.data_reader = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.setThermo()
        self.analyze()

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return

        self.data_reader = stillinger.get_data_Reader(self.options.data_file)
        self.data_reader.run()

    def setThermo(self):
        """
        Grep thermo output information.
        """

        self.lmp_log = fileutils.LammpsLog(self.options.lmp_log,
                                           last_pct=self.options.last_pct)
        self.lmp_log.run()

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
                f"[{self.lmp_log.thermo.index[self.lmp_log.sidx]: .3f}, "
                f"{self.lmp_log.thermo.index[-1]: .3f}] ps")

    def analyze(self):
        """
        Run analyzers.
        """

        thermo_tasks = [x for x in THERMO_TASKS if x in self.options.task]
        if thermo_tasks:
            filename = self.options.jobname + self.DATA_EXT % THERMO
            self.lmp_log.write(thermo_tasks, filename)
            log(f'{thermo_tasks} info written into {filename}')

    @classmethod
    def getOutfiles(cls, logfile):
        """
        Get the output files based on the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :return: task name and the related outfile
        :rtype: dict
        """
        jobname = cls.getLogged(logfile)[0]
        tsks = cls.getLogged(logfile, key=cls.TASK, strip='[]', delimiter=', ')
        thermo_filename = jobname + cls.DATA_EXT % THERMO
        return {
            x: thermo_filename if x in THERMO_TASKS else jobname +
            cls.DATA_EXT % x
            for x in tsks
        }

    @classmethod
    def getLogged(cls, logfile, key=None, strip=None, delimiter=None):
        """
        Get the values corresponding to the key in the log file.

        :param logfile: the log file generated by this class
        :type logfile: str
        :param key: the key based on which values are fetched
        :type key: str
        :param delimiter: the chars to strip the string
        :type delimiter: str
        :param delimiter: the delimiter to split the string
        :type delimiter: str
        :return: the matching values in the logfile
        :rtype: list
        """
        if key is None:
            key = jobutils.FLAG_JOBNAME.lower()[1:]
        block = sh.grep(f'{key}:', logfile)
        matched = re.findall(f"(?<={key[1:]}: ).+(?=\n)", block)[0]
        matched = matched.strip(strip)
        matched = [x.strip() for x in matched.split(delimiter)]
        return matched

    @classmethod
    def combine(cls, files, log, name, inav=False):
        """
        Concatenate multiple outfiles from the same task into one.

        :param files: task name and the related outfile
        :type files: dict
        :param log: the function to print user-facing information
        :type log: 'function'
        :param name: output files are named based on this name
        :type name: str
        :param inav: pop up window and show plot during code execution if
            interactive mode is on
        :type inav: bool
        """

        for aname, afiles in files.items():
            if aname in THERMO_TASKS:
                continue
            if aname in NO_COMBINE:
                continue
            import pdb
            pdb.set_trace()

            Analyzer = cls.ANALYZER[aname]
            data = Analyzer.read(name, files=afiles, log=log)
            sidx, eidx = Analyzer.fit(data, log=log)
            Analyzer.plot(data, name, inav=inav, sidx=sidx, eidx=eidx, log=log)


def get_parser(parser=None):
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    if parser is None:
        parser = parserutils.get_parser(description=__doc__)
        parser.add_argument(FlAG_LMP_LOG,
                            metavar=FlAG_LMP_LOG.upper(),
                            type=parserutils.type_file,
                            help='LAMMPS log file to analyze')
        parser.add_argument(FlAG_DATA_FILE,
                            metavar=FlAG_DATA_FILE[1:].upper(),
                            type=parserutils.type_file,
                            help='Data file to get force field information')
    parser.add_argument(FlAG_TASK,
                        choices=THERMO_TASKS,
                        default=THERMO_TASKS,
                        nargs='+',
                        help=f'{THERMO} searches, combines and averages '
                        f'thermodynamic info. ')
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
        parser.error(f"Please specify {FlAG_DATA_FILE} to run {FlAG_TASK} "
                     f"{', '.join(data_rqd_tasks)}")

    try:
        options.task.remove(analyzer.XYZ.NAME)
    except ValueError:
        pass
    else:
        # XYZ analyzer may change the coordinates
        options.task.append(analyzer.XYZ.NAME)
    return options


logger = None


def main(argv):

    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    lmp_log = LmpLog(options)
    lmp_log.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
