# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow driver runs crystal builder, lammps, and log analyser.
"""
import os
import sh
import sys
import numpy as np
import pandas as pd
from flow import FlowProject

from nemd import logutils
from nemd import stillinger
from nemd import jobutils
from nemd import jobcontrol
from nemd import parserutils
from nemd import environutils
from nemd.task import Crystal_Builder, Lammps, Lmp_Log

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

FLAG_SCALED_RANGE = '-scaled_range'


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
def label(job):
    """
    Show the label of a job (job id is a long alphanumeric string).

    :param job 'signac.contrib.job.Job': the job object
    :return str: the job name of a subtask
    """

    return str(job.statepoint())


class Runner(jobcontrol.Runner):

    MINIMUM_ENERGY = "yields the minimum energy of"
    LMP_LOG = 'lmp_log'

    def setTasks(self):
        """
        Set polymer builder, lammps builder, and custom dump tasks.
        """
        crystal_builder = Crystal_Builder.getOpr(name='crystal_builder')
        lammps_runner = Lammps.getOpr(name='lammps_runner')
        self.setPrereq(lammps_runner, crystal_builder)
        lmp_log = Lmp_Log.getOpr(name='lmp_log')
        self.setPrereq(lmp_log, lammps_runner)

    def addJobs(self):
        """
        Add jobs to the project.
        """
        ids = np.arange(*self.options.scaled_range)
        super().addJobs(ids=ids)

    def setAggregation(self):
        """
        Aggregate post analysis jobs.
        """
        super().setAggregation()
        name = f"{self.options.jobname}{self.SEP}{self.LMP_LOG}"
        combine_agg = Lmp_Log.getAgg(name=name,
                                     tname=self.LMP_LOG,
                                     log=log,
                                     clean=self.options.clean,
                                     state_label='Scale Factor')
        name = f"{self.options.jobname}{self.SEP}fitting"
        fit_agg = Lmp_Log.getAgg(name=name,
                                 attr=self.minEneAgg,
                                 tname=Lmp_Log.DRIVER.TOTENG,
                                 post=self.minEnePost,
                                 log=log,
                                 clean=self.options.clean,
                                 state_label='Scale Factor')
        self.setPrereq(fit_agg, combine_agg)

    @staticmethod
    def minEneAgg(*jobs, log=None, name=None, tname=None, **kwargs):
        """
        The aggregator job task that combines the output files of a custom dump
        task.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param log: the function to print user-facing information
        :type log: 'function'
        :param name: the jobname based on which output files are named
        :type name: str
        :param tname: aggregate the job tasks of this name
        :type tname: str
        """
        jname = name.split(Runner.SEP)[0]
        filename = jname + Lmp_Log.DRIVER.LmpLog.AVE_DATA_EXT % Lmp_Log.DRIVER.THERMO
        data = pd.read_csv(filename, index_col=0)
        columns = [x for x in data.columns if x.split('(')[0].strip() == tname]
        index = data[columns[0]].argmin()
        factor = data.iloc[index].name
        val = data.iloc[index][columns[0]]
        unit = columns[0].split('(')[-1].split(')')[0]
        log(f"A scale factor of {factor} {Runner.MINIMUM_ENERGY} {val} {unit}")
        job = [x for x in jobs if x.statepoint[jobutils.STATE_ID] == factor][0]
        datafile = [
            x for x in job.doc[jobutils.OUTFILES]['crystal_builder']
            if x.endswith(stillinger.Struct.DATA_EXT)
        ][0]
        log(f'The corresponding datafile is saved as in {datafile}')

    @classmethod
    def minEnePost(cls, *jobs, name=None):
        """
        Report the status of the aggregation for minimum energy.

        Main driver log should report results found the csv saved on the success
        of aggregation.

        :param jobs: the task jobs the aggregator collected
        :type jobs: list of 'signac.contrib.job.Job'
        :param name: jobname based on which log file is found
        :type name: str
        :return: the label after job completion
        :rtype: str
        """
        jname = name.split(cls.SEP)[0]
        logfile = jname + logutils.DRIVER_LOG
        try:
            line = sh.grep(cls.MINIMUM_ENERGY, logfile)
        except sh.ErrorReturnCode_1:
            return False
        return line.split(cls.MINIMUM_ENERGY)[0].strip()


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser': argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(
        FLAG_SCALED_RANGE,
        default=(0.95, 1.05, 0.01),  # yapf: disable
        nargs='+',
        metavar=FLAG_SCALED_RANGE.upper()[1:],
        type=parserutils.type_positive_float,
        help='Number of states for the dynamical system via random seed')
    parser = Crystal_Builder.DRIVER.get_parser(parser)
    parser = Lmp_Log.DRIVER.get_parser(parser)
    parserutils.add_job_arguments(parser,
                                  jobname=environutils.get_jobname(JOBNAME))
    parserutils.add_workflow_arguments(parser)
    return parser


def validate_options(argv):
    """
    Parse and validate the command options.

    :param argv list: command arguments
    :return 'argparse.Namespace': parsed command line options.
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    if Lmp_Log.DRIVER.TOTENG not in options.task:
        options.task += [Lmp_Log.DRIVER.TOTENG]
        index = argv.index(Lmp_Log.DRIVER.FLAG_TASK)
        argv.insert(index + 1, Lmp_Log.DRIVER.TOTENG)
    return options


logger = None


def main(argv):
    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname)
    logutils.logOptions(logger, options)
    runner = Runner(options, argv, logger=logger)
    runner.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log('finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
