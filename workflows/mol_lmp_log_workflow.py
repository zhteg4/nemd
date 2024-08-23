# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow driver runs polymer builder, lammps, and log analyser.
"""
import os
import sys
import numpy as np
from flow import FlowProject

from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import polymutils
from nemd import jobcontrol
from nemd import parserutils
from nemd.task import MolBldr, Lammps, LmpLog

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_workflow', '')

FLAG_SUBSTRUCT_RANGE = '-substruct_range'


def log(msg, timestamp=False):
    """
    Print this message into log file in regular mode.

    :param msg: the msg to print
    :param timestamp bool: the msg to be printed
    """
    if not logger:
        return
    logutils.log(logger, msg, timestamp=timestamp)


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

    def setJob(self):
        """
        Set crystal builder, lammps runner, and log analyzer tasks.
        """
        conformer_builder = MolBldr.getOpr(name='conformer_builder')
        lammps_runner = Lammps.getOpr(name='lammps_runner')
        self.setPrereq(lammps_runner, conformer_builder)
        lmp_log = LmpLog.getOpr(name='lmp_log')
        self.setPrereq(lmp_log, lammps_runner)

    def setState(self):
        """
        Set the state keys and values.
        """
        super().setState()
        if self.options.substruct_range[1] is None:
            self.state[polymutils.FLAG_SUBSTRUCT] = [
                self.options.substruct_range[0]
            ]
            return
        range_values = map(str, np.arange(*self.options.substruct_range[1]))
        substruct = self.options.substruct[0]
        structs = [symbols.COLON.join([substruct, x]) for x in range_values]
        self.state[polymutils.FLAG_SUBSTRUCT] = structs

    def setAggregation(self):
        """
        Aggregate post analysis jobs.
        """
        super().setAggregation()
        name = f"{self.options.jobname}{self.SEP}{self.LMP_LOG}"
        combine_agg = LmpLog.getAgg(name=name,
                                    tname=self.LMP_LOG,
                                    log=log,
                                    clean=self.options.clean,
                                    state_label='Scale Factor')
        name = f"{self.options.jobname}{self.SEP}fitting"
        fit_agg = LmpLog.getAgg(name=name,
                                attr=self.minEneAgg,
                                tname=LmpLog.DRIVER.TOTENG,
                                post=self.minEnePost,
                                log=log,
                                clean=self.options.clean,
                                state_label='Scale Factor')
        self.setPrereq(fit_agg, combine_agg)


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser': argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(
        FLAG_SUBSTRUCT_RANGE,
        metavar='SMILES:START,END,STEP',
        type=lambda x: parserutils.type_substruct(x, is_range=True),
        help='The range of the degree to scan in degrees. ')
    parser = MolBldr.DRIVER.get_parser(parser)
    parser = LmpLog.DRIVER.get_parser(parser)
    parser.supress_arguments([
        parserutils.FLAG_LAST_PCT, parserutils.FLAG_SLICE,
        parserutils.FLAG_STATE_NUM
    ])
    parserutils.add_job_arguments(parser, jobname=JOBNAME)
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
    if options.substruct_range is None:
        parser.error(f"Please specify the substructure and scanning range "
                     f"using {FLAG_SUBSTRUCT_RANGE} option.")
    return options


logger = None


def main(argv):
    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=JOBNAME, set_file=True)
    logutils.logOptions(logger, options)
    runner = Runner(options, argv, logger=logger)
    runner.run()
    log(jobutils.FINISHED, timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
