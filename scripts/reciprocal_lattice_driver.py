# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (zhteg4@gmail.com)

# https://www.youtube.com/watch?v=cdN6OgwH8Bg
# https://en.wikipedia.org/wiki/Reciprocal_lattice
"""
Calculate and visualize hexagonal 2D lattice in real and reciprocal spaces.
"""
import os
import sys
import math
import numpy as np

from nemd import jobutils
from nemd import logutils
from nemd import plotutils
from nemd import parserutils
from nemd import environutils

PATH = os.path.basename(__file__)
JOBNAME = PATH.split('.')[0].replace('_driver', '')

FLAG_MILLER_INDICES = '-miller_indices'


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


class Reciprocal:
    PNG_EXT = '.png'

    def __init__(self, options):
        self.options = options
        self.origin = np.array([0., 0.])

    def run(self):
        self.setRealVectors()
        self.setReciprocalVectors()
        self.plot()

    def setRealVectors(self):
        """
        Define real space lattice vector with respect to the origin.
        """
        # characteristic_length
        # https://physics.stackexchange.com/questions/664945/integration-over-first-brillouin-zone
        charac_length = math.sqrt(3)  # sqrt(3) X the edge length of hexagon
        self.a_pnt = np.array([math.sqrt(3) / 2., 0.5]) * charac_length
        self.b_pnt = np.array([math.sqrt(3) / 2., -0.5]) * charac_length
        self.a_vect = self.a_pnt - self.origin
        self.b_vect = self.b_pnt - self.origin

    def setReciprocalVectors(self):
        """
        Set the reciprocal lattice vectors based on the real ones.
        """
        self.ga_vect = self.getGVector(self.a_vect, self.b_vect)
        self.gb_vect = self.getGVector(self.b_vect, self.a_vect)

    @staticmethod
    def getGVector(a_vect, b_vect):
        """
        return the reciprocal vector of a vector with respect to b vector in 2D.

        :param a_vect: one real vector in 2D space
        :param b_vect: the other real vector in 2D space
        :return:
        """
        vertical = np.dot([[0, -1], [1, 0]], b_vect)
        return 2 * np.pi * vertical / np.dot(a_vect, vertical)

    def plot(self):
        """
        Plot the real and reciprocal paces.
        """
        with plotutils.get_pyplot() as plt:
            fig = plt.figure(figsize=(15, 9))
            self.ax1 = fig.add_subplot(1, 2, 1)
            self.ax2 = fig.add_subplot(1, 2, 2)
            ltp = plotutils.LatticePlotter(self.ax1,
                                           a_vect=self.a_vect,
                                           b_vect=self.b_vect,
                                           indices=self.options.miller_indices)
            ltp.run()
            log(f"The vector in the real space is: {ltp.vect} with "
                f"{np.linalg.norm(ltp.vect)} being the norm.")
            rltp = plotutils.ReciprocalLatticePlotter(
                self.ax2,
                a_vect=self.ga_vect,
                b_vect=self.gb_vect,
                indices=self.options.miller_indices)
            rltp.run()
            log(f"The vector in the reciprocal space is: {rltp.vect} with "
                f"{np.linalg.norm(rltp.vect)} being the norm.")
            log(f"The cross product is {np.cross(ltp.vect, rltp.vect): .4g}")
            log(f"The product is {np.dot(ltp.vect, rltp.vect) / np.pi: .4g} * pi"
                )
            fig.tight_layout()
            if self.options.interactive:
                print(f"Showing the plot. Click X to close and continue..")
                plt.show(block=True)
            fname = self.options.jobname + self.PNG_EXT
            fig.savefig(fname)
            jobutils.add_outfile(fname, jobname=self.options.jobname)
        log(f'Figure saved as {fname}')


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
    parserutils.add_job_arguments(parser,
                                  arg_flags=jflags,
                                  jobname=environutils.get_jobname(JOBNAME))
    parser.add_argument(FLAG_MILLER_INDICES,
                        metavar=FLAG_MILLER_INDICES[1:].upper(),
                        default=[0.5, 2],
                        type=parserutils.type_int,
                        nargs='+',
                        help='Plot the planes of this Miller indices .')
    return parser


def validate_options(argv):
    """
    Parse and validate the command args

    :param argv list: list of command input.
    :return: 'argparse.ArgumentParser':  Parsed command-line options out of sys.argv
    """
    parser = get_parser()
    options = parser.parse_args(argv)
    if len(options.miller_indices) != 2:
        parser.error(
            f'Please provide two integers as the Miller indices ({FLAG_MILLER_INDICES}).'
        )
    if not np.array(options.miller_indices).any():
        parser.error(
            f'Miller indices cannot be all zeros ({FLAG_MILLER_INDICES}).')
    return options


logger = None


def main(argv):

    global logger
    options = validate_options(argv)
    logger = logutils.createDriverLogger(jobname=options.jobname,
                                         log_file=True)
    logutils.logOptions(logger, options)
    reciprocal = Reciprocal(options)
    reciprocal.run()
    log_file = os.path.basename(logger.handlers[0].baseFilename)
    jobutils.add_outfile(log_file, options.jobname, set_file=True)
    log('Finished.', timestamp=True)


if __name__ == "__main__":
    main(sys.argv[1:])
