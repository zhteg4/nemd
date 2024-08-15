"""
This runs lammps executable with the given input file and output file.
"""
import os
import re
import sh
import sys
import subprocess

from nemd import symbols
from nemd import jobutils
from nemd import logutils
from nemd import lammpsin
from nemd import parserutils
from nemd import environutils

FLAG_INSCRIPT = 'inscript'
FLAG_SCREEN = '-screen'
FLAG_LOG = '-log'
FLAG_DATA_FILE = parserutils.FLAG_DATA_FILE

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

# Positional command-argument holders to take task input under jobcontrol
ARGS_TMPL = [jobutils.FILE]


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

    FLAG_IN = '-in'
    LMP_SERIAL = 'lmp_serial'
    GREP_GPU = f'{LMP_SERIAL} -h | grep GPU'
    READ_DATA = re.compile(f'^{lammpsin.In.READ_DATA} +(.*)$')
    GREP_DATA = f'grep {lammpsin.In.READ_DATA} {{inscript}}'
    LOG = '.log'

    def __init__(self, options):
        self.options = options
        self.args = []
        self.outfile = self.options.jobname + self.LOG
        jobutils.add_outfile(self.outfile,
                             jobname=self.options.jobname,
                             set_file=True)

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
        self.args += [FLAG_LOG, self.outfile, FLAG_SCREEN, self.options.screen]
        info = sh.grep(lammpsin.In.READ_DATA, self.options.inscript)
        if not info:
            return

        inscript = os.path.basename(self.options.inscript)
        data_file = Lammps.READ_DATA.match(info).groups()[0]
        if os.path.isfile(inscript) and os.path.exists(data_file):
            return
        with open(self.options.inscript, 'r') as fh:
            original_contents = fh.read()
            read_data = f'{lammpsin.In.READ_DATA} {self.options.data_file}'
            contents = original_contents.replace(info, read_data)
        with open(inscript, 'w') as fh:
            fh.writelines(contents)
        self.options.inscript = inscript

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
        cmd = [self.LMP_SERIAL, '-in', self.options.inscript] + self.args
        cmd = symbols.SPACE.join(cmd)
        info = subprocess.run(cmd, capture_output=True, shell=True)
        if info.stderr:
            log('WARNING:' + info.stderr.decode('utf-8'))
        if info.stdout:
            log(info.stdout.decode('utf-8'))
        # https://docs.lammps.org/jump.html
        # The SELF option is not guaranteed to work when the current input
        # script is being read through stdin (standard input), e.g.
        # with logutils.redirect(logger=logger):
        #     # "[xxx.local:xxx] shmem: mmap: an error occurred while determining
        #     # whether or not xxx could be created." while lammps.lammps()
        #     lmp = lammps.lammps(cmdargs=self.args)
        # with open(self.options.inscript, 'r') as fh:
        #     cmds = fh.readlines()
        #     rdata = f'{lammpsin.In.READ_DATA} {self.options.data_file}'
        #     cmds = [rdata if self.READ_DATA.match(x) else x for x in cmds]
        #     breakpoint()
        #     lmp.commands_list(cmds)
        # lmp.close()


def get_parser(parser=None):
    """
    Get the customized parser wrapper for lammps executable.

    :return: the customized parser wrapper
    :rtype: 'argparse.ArgumentParser'
    """
    if parser is None:
        parser = parserutils.get_parser(
            description='This is a customized parser wrapper for lammps.')
        parser.add_argument(FLAG_INSCRIPT,
                            metavar=FLAG_INSCRIPT.upper(),
                            type=parserutils.type_file,
                            help='Read input from this file.')
    parser.add_argument(FLAG_SCREEN,
                        metavar=FLAG_SCREEN[1:].upper(),
                        choices=[symbols.NONE, symbols.FILENAME],
                        default=symbols.NONE,
                        help='Where to send screen output (-sc)')
    parser.add_argument(FLAG_LOG,
                        metavar=FLAG_LOG[1:].upper(),
                        help='Print logging information into this file.')
    parser.add_argument(FLAG_DATA_FILE,
                        metavar=FLAG_DATA_FILE[1:].upper(),
                        type=parserutils.type_file,
                        help='Data file to get force field information')
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
    return parser


class Validator:

    def __init__(self, options):
        """
        param options: Command line options.
        """
        self.options = options

    def run(self):
        """
        Main method to run the validation.
        """
        self.dataFile()

    def dataFile(self):
        """
        Check if the data file exists.

        :raises FileNotFoundError: if the data file is required but does not
            exist.
        """
        if self.options.data_file:
            return

        cmd = Lammps.GREP_DATA.format(inscript=self.options.inscript)
        info = subprocess.run(cmd, capture_output=True, shell=True)
        if not info.stdout:
            return

        file = Lammps.READ_DATA.match(info.stdout.decode('utf-8')).groups()[0]
        if not os.path.isfile(file):
            dirname = os.path.dirname(self.options.inscript)
            file = os.path.join(dirname, file)

        if not os.path.isfile(file):
            raise FileNotFoundError(f'Data file {file} does not exist.')
        self.options.data_file = file


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    validator = Validator(options)
    try:
        validator.run()
    except (FileNotFoundError, ValueError) as err:
        parser.error(err)
    return validator.options
    return options


logger = None


def main(argv):
    global logger

    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=JOBNAME)
    logutils.logOptions(logger, options)
    lmp = Lammps(options)
    lmp.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
