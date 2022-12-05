import os
import sys
import traj
import oplsua
import jobutils
import logutils
import parserutils
import environutils
import numpy as np
import pandas as pd

FlAG_CUSTOM_DUMP = 'custom_dump'
FlAG_DATA_FILE = '-data_file'
FlAG_TASK = '-task'
CLASH = 'clash'
XYZ = 'xyz'

JOBNAME = os.path.basename(__file__).split('.')[0].replace('_driver', '')


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


def get_parser():
    """
    The user-friendly command-line parser.

    :return 'argparse.ArgumentParser':  argparse figures out how to parse those
        out of sys.argv.
    """
    parser = parserutils.get_parser(
        description='Perform tasks on customer dump')
    parser.add_argument(FlAG_CUSTOM_DUMP,
                        metavar=FlAG_CUSTOM_DUMP.upper(),
                        type=parserutils.type_file,
                        help='Custom dump file to analyze')
    parser.add_argument(FlAG_DATA_FILE,
                        metavar=FlAG_DATA_FILE.upper(),
                        type=parserutils.type_file,
                        help='Data file to get force field information')
    parser.add_argument(FlAG_TASK,
                        choices=[XYZ, CLASH],
                        default=[XYZ],
                        nargs='+',
                        help=f'{XYZ} writes out .xyz for VMD visualization;'
                        f'{CLASH} check clashes for each frame.')

    jobutils.add_job_arguments(parser)
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)

    if CLASH in options.task and not options.data_file:
        parser.error(
            f'Please specify {FlAG_DATA_FILE} to run {FlAG_TASK} {CLASH}')
    return options


class CustomDump(object):

    XYZ_EXT = '.xyz'

    def __init__(self, options, jobname, diffusion=False):
        """
        :param options 'argparse.ArgumentParser': Parsed command-line options
        :param jobname str: jobname of this task
        :param diffusion bool: particles passing PBCs continue traveling
        """
        self.options = options
        self.jobname = jobname
        self.diffusion = diffusion
        self.outfile = self.jobname + self.XYZ_EXT
        self.data_reader = None
        self.radii = None

    def run(self):
        """
        Main method to run the tasks.
        """
        self.setStruct()
        self.checkClashes()
        self.writeXYZ()
        log('Finished', timestamp=True)

    def setStruct(self):
        """
        Load data file and set clash parameters.
        """
        if not self.options.data_file:
            return

        self.data_reader = oplsua.DataFileReader(self.options.data_file)
        self.data_reader.run()
        self.data_reader.setClashParams()

    def checkClashes(self):
        """
        Check clashes for reach frames.
        """
        if CLASH not in self.options.task:
            return

        for idx, frm in enumerate(traj.Frame.read(self.options.custom_dump)):
            clashes = self.getClashes(frm)
            log(f"Frame {idx} has {len(clashes)} clashes.")
        log('All frames are checked for clashes.')

    def getClashes(self, frm):
        """
        Get the clashes between atom pair for this frame.

        :param frm 'traj.Frame':
        :return list of tuples: each tuple has two atom ids, the distance, and
            clash threshold
        """
        clashes = []
        dcell = traj.DistanceCell(frm=frm)
        dcell.setUp()
        for _, row in frm.iterrows():
            clashes += dcell.getClashes(row,
                                        radii=self.data_reader.radii,
                                        excluded=self.data_reader.excluded)
        return clashes

    def writeXYZ(self, wrapped=True, bond_across_pbc=False, glue=True):
        if glue and not (wrapped and bond_across_pbc is False):
            raise ValueError(f'Glue moves molecules together like droplets.')

        if XYZ not in self.options.task:
            return

        with open(self.outfile, 'w') as self.out_fh:
            for frm in traj.Frame.read(self.options.custom_dump):
                self.wrapCoords(frm,
                                wrapped=wrapped,
                                bond_across_pbc=bond_across_pbc,
                                glue=glue)
                self.out_fh.write(f'{frm.shape[0]}\n')
                if self.data_reader:
                    index = [self.data_reader.atoms[x].ele for x in frm.index]
                else:
                    index = ['X'] * frm.shape[0]
                frm.index = index
                frm.to_csv(self.out_fh,
                           mode='a',
                           index=True,
                           sep=' ',
                           header=True)
        log(f"Coordinates are written into {self.outfile}")

    def wrapCoords(self, frm, wrapped=True, bond_across_pbc=False, glue=True):
        if not wrapped:
            return

        box = frm.attrs['box']
        span = np.array([box[i * 2 + 1] - box[i * 2] for i in range(3)])
        if bond_across_pbc:
            frm = frm % span
            return

        if not self.data_reader:
            return

        for mol in self.data_reader.mols.values():
            center = frm.loc[mol].mean()
            delta = (center % span) - center
            frm.loc[mol] += delta

        if not glue:
            return

        centers = pd.concat(
            [frm.loc[x].mean() for x in self.data_reader.mols.values()],
            axis=1).transpose()
        centers.index = self.data_reader.mols.keys()
        theta = centers / span * 2 * np.pi
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta = np.arctan2(sin_theta.mean(), cos_theta.mean())
        mcenters = theta * span / 2 / np.pi
        cshifts = ((mcenters - centers) / span).round()
        for id, mol in self.data_reader.mols.items():
            cshift = cshifts.loc[id]
            frm.loc[mol] += cshift * span


logger = None


def main(argv):
    global logger

    jobname = environutils.get_jobname(JOBNAME)
    logger = logutils.createDriverLogger(jobname=jobname)
    options = validate_options(argv)
    logutils.logOptions(logger, options)
    cdump = CustomDump(options, jobname)
    cdump.run()


if __name__ == "__main__":
    main(sys.argv[1:])
