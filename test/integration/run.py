import os
import sys
import glob
import filecmp
import contextlib
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import fileutils
from nemd import parserutils
from nemd import environutils
from nemd.nflow import FlowProject

DIR = 'dir'
CHECKED = 'checked'
CMD = 'cmd'
JOBNAME = 'integration_test'

FLAG_DIR = 'dir'


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
    :param timestamp bool: the msg to be printed
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


@FlowProject.label
def cmd_completed(job):
    return len(glob.glob(job.fn(symbols.WILD_CARD))) > 1


@FlowProject.post(cmd_completed)
@FlowProject.operation(cmd=True)
def run_cmd(job):
    test_cmd_file = os.path.join(job.document[DIR], CMD)
    with open(test_cmd_file) as fh:
        lines = [x.strip() for x in fh.readlines()]
    cmd = symbols.SEMICOLON.join(lines)
    return f"cd {job.path}; {cmd}; cd -"


def checked(job):
    return CHECKED in job.document


@FlowProject.pre.after(run_cmd)
@FlowProject.post(cmd_completed)
@FlowProject.operation
def check(job):
    assert filecmp.cmp('polymer_builder.data', job.fn('polymer_builder.data'))
    job.document[CHECKED] = True


class Integration:
    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname
        self.test_dirs = None
        self.project = None
        self.status_file = self.jobname + fileutils.STATUS_LOG
        logutils.createModuleLogger('flow.project', file_ext='.log')
        self.status_fh = None

    def run(self):
        with open(self.status_file, 'w') as self.status_fh:
            self.setTests()
            self.setProject()
            self.addJobs()
            self.runProject()
            self.logStatus()

    def setTests(self):
        base_dir = os.path.join(self.options.dir, symbols.WILD_CARD)
        self.test_dirs = [
            x for x in glob.glob(base_dir)
            if os.path.isdir(x) and os.path.basename(x).isdigit()
        ]
        if not self.test_dirs:
            log_error(f'No tests found in {self.options.dir}.')
        log(f"{len(self.test_dirs)} tests found.")

    def setProject(self, workspace='workspace'):
        self.project = FlowProject.init_project(workspace=workspace)

    def addJobs(self):
        for test_dir in self.test_dirs:
            job = self.project.open_job({'id': os.path.basename(test_dir)})
            job.document[DIR] = test_dir
            job.init()

    def runProject(self):
        self.project.run()

    def logStatus(self):
        self.project.print_status(detailed=True,
                                  file=self.status_fh,
                                  err=self.status_fh)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(
        description='Build amorphous cell from molecules and monomers.')
    parser.add_argument(FLAG_DIR,
                        metavar=FLAG_DIR.upper(),
                        type=parserutils.type_dir,
                        nargs='?',
                        help='The directory to search for integration tests.')
    jobutils.add_job_arguments(parser)
    return parser


def validate_options(options):
    if not options.dir:
        options.dir = environutils.get_integration_test_dir()
    return options


logger = None


def main(argv):
    global logger

    parser = get_parser()
    options = parser.parse_args(argv)
    try:
        options = validate_options(options)
    except ValueError as err:
        parser.error(str(err))

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    logutils.logOptions(logger, options)
    integration = Integration(options, jobname)
    integration.run()


if __name__ == "__main__":
    main(sys.argv[1:])
