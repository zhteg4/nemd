"""
This runs lammps executable with the given input file and output file.
"""
import os
import sys
import lammps
import subprocess

from nemd import symbols
from nemd import jobutils
from nemd import logutils
from nemd import parserutils
from nemd import environutils


FLAG_INSCRIPT = '-inscript'
FLAG_SCREEN = '-screen'
FLAG_LOG = '-log'

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

# Positional command-argument holders to take task input under jobcontrol
ARGS_TMPL = [FLAG_INSCRIPT, jobutils.FILE]


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: append time information after the message
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


class Lammps:

    LMP_SERIAL = 'lmp_serial'
    GREP_GPU = f'{LMP_SERIAL} -h | grep GPU'
    LOG = '.log'

    def __init__(self, options):
        self.options = options
        self.args = []

    def run(self):
        """
        Run lammps executable with the given input file and output file.
        """
        self.setArgs()
        self.setGpu()
        self.runLammps()

    def setArgs(self):
        """
        Set the arguments for the lammps executable.
        """
        file = self.options.jobname + self.LOG
        self.args += [FLAG_LOG, file, FLAG_SCREEN, self.options.screen]
        jobutils.add_outfile(file, jobname=self.options.jobname, set_file=True)

    def setGpu(self):
        """
        Set the arguments for the GPU lammps executable.
        """
        lmp = subprocess.run(self.GREP_GPU, capture_output=True, shell=True)
        if not lmp.stdout:
            return
        self.args += ['-sf', 'gpu', '-pk', 'gpu', '1']

    def runLammps(self):
        """
        Run lammps executable with the given input file and output file.
        """
        log('Running lammps simulations...')
        lmp = lammps.lammps(cmdargs=self.args)
        lmp.file(self.options.inscript)
        lmp.close()


def get_parser():
    """
    Get the customized parser wrapper for lammps executable.

    :return: the customized parser wrapper
    :rtype: 'argparse.ArgumentParser'
    """
    parser = parserutils.get_parser(
        description='This is a customized parser wrapper for lammps.')
    parser.add_argument(FLAG_INSCRIPT,
                        metavar=FLAG_INSCRIPT[1:].upper(),
                        type=parserutils.type_file,
                        required=True,
                        help='Read input from this file.')
    parser.add_argument(FLAG_SCREEN,
                        metavar=FLAG_SCREEN[1:].upper(),
                        choices=[symbols.NONE, symbols.FILENAME],
                        default=symbols.NONE,
                        help='Where to send screen output (-sc)')
    parser.add_argument(FLAG_LOG,
                        metavar=FLAG_LOG[1:].upper(),
                        help='Print logging information into this file.')
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
    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    lmp = Lammps(options)
    lmp.run()
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])