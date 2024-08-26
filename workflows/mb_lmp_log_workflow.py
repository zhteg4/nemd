# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)
"""
This workflow driver runs molecule builder, lammps, and log analyzer.
"""
import os
import sys
import rdkit
import numpy as np
from flow import FlowProject

from nemd import task
from nemd import symbols
from nemd import logutils
from nemd import jobutils
from nemd import analyzer
from nemd import polymutils
from nemd import jobcontrol
from nemd import parserutils
from nemd.task import MolBldr, Lammps, LmpLog

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_workflow', '')

FLAG_STRUCT_RG = '-struct_rg'


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


class LogReader(logutils.LogReader):

    def getSubstruct(self, smiles):
        """
        Get the value of a substructure from the log file.

        :param smiles str: the substructure smiles
        :return str: the value of the substructure
        """
        for line in self.lines[self.sidx:]:
            if not line.startswith(smiles):
                continue
            # e.g. 'CCCC dihedral angle: 73.50 deg'
            return line.split(symbols.COLON)[-1].split()[0]


class AnalyzerAgg(analyzer.Agg):

    SUBSTRUCT = polymutils.FLAG_SUBSTRUCT[1:].capitalize()

    def setVals(self):
        """
        Set the xvals name in addition to the default behavior.
        """
        super().setVals()
        vals = self.xvals[self.SUBSTRUCT].str.split(symbols.COLON, expand=True)
        smiles = vals.iloc[0, 0]
        match rdkit.Chem.MolFromSmiles(smiles).GetNumAtoms():
            case 2:
                name = f"{smiles} Bond (Angstrom)"
            case 3:
                name = f"{smiles} Angle (Degree)"
            case 4:
                name = f"{smiles} Dihedral Angle (Degree)"
        self.xvals = self.xvals.rename(columns={self.SUBSTRUCT: name})
        try:
            self.xvals.loc[:, name] = vals[1]
        except IndexError:
            self.xvals.loc[:, name] = self.getSubstruct(smiles)

    def getSubstruct(self, smiles):
        """
        Get the value of a substructure from the log file.

        :param smiles str: the substructure smiles
        :return str: the value of the substructure
        """
        job = self.jobs[0][-1][0]
        for logfile in job.doc[jobutils.LOGFILE].values():
            reader = LogReader(job.fn(logfile))
            if reader.options.default_name != MolBldr.DRIVER.JOBNAME:
                continue
            return reader.getSubstruct(smiles)


class LogJobAgg(task.LogJobAgg):

    AnalyzerAgg = AnalyzerAgg


class LmpLog(LmpLog):

    AggClass = LogJobAgg


class Runner(jobcontrol.Runner):

    MINIMUM_ENERGY = "yields the minimum energy of"
    LMP_LOG = 'lmp_log'

    def setJob(self):
        """
        Set molecule builder, lammps runner, and log analyzer tasks.
        """
        mol_bldr = self.setOpr(MolBldr)
        lmp_runner = self.setOpr(Lammps)
        self.setPreAfter(mol_bldr, lmp_runner)
        lmp_log = self.setOpr(LmpLog)
        self.setPreAfter(lmp_runner, lmp_log)

    def setState(self):
        """
        Set the state keys and values.
        """
        super().setState()
        if self.options.struct_rg[1] is None:
            self.state[polymutils.FLAG_SUBSTRUCT] = [self.options.struct_rg[0]]
            return
        range_values = map(str, np.arange(*self.options.struct_rg[1:]))
        substruct = self.options.struct_rg[0]
        structs = [symbols.COLON.join([substruct, x]) for x in range_values]
        self.state[polymutils.FLAG_SUBSTRUCT] = structs

    def setAggJobs(self):
        """
        Aggregate post analysis jobs.
        """
        self.setAgg(LmpLog)
        super().setAggJobs()


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser': argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(description=__doc__)
    parser.add_argument(
        FLAG_STRUCT_RG,
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
    if options.struct_rg is None:
        parser.error(f"Please specify the substructure and scanning range "
                     f"using {FLAG_STRUCT_RG} option.")
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
