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
        self.meshRealSpace()
        self.setReciprocalVectors()
        self.plot()

    def setRealVectors(self):
        # characteristic_length
        # https://physics.stackexchange.com/questions/664945/integration-over-first-brillouin-zone
        charac_length = math.sqrt(3)  # sqrt(3) X the edge length of hexagon
        self.a_pnt = np.array([math.sqrt(3) / 2., 0.5]) * charac_length
        self.b_pnt = np.array([math.sqrt(3) / 2., -0.5]) * charac_length
        self.a_vect = self.a_pnt - self.origin
        self.b_vect = self.b_pnt - self.origin

    def meshRealSpace(self, num=10):
        xv, yv = np.meshgrid(range(-num, num + 1), range(-num, num + 1))
        self.real_xs = xv * self.a_vect[0] + yv * self.b_vect[0]
        self.real_ys = xv * self.a_vect[1] + yv * self.b_vect[1]

    def plotRealGrids(self):
        bottom_idx = np.unravel_index(self.real_ys.argmin(),
                                      self.real_ys.shape)
        top_idx = np.unravel_index(self.real_ys.argmax(), self.real_ys.shape)
        left_idx = np.unravel_index(self.real_xs.argmin(), self.real_xs.shape)
        right_idx = np.unravel_index(self.real_xs.argmax(), self.real_xs.shape)
        tl_x = (self.real_xs[top_idx] + self.real_xs[left_idx]) / 2
        tl_y = (self.real_ys[top_idx] + self.real_ys[left_idx]) / 2
        tr_x = (self.real_xs[top_idx] + self.real_xs[right_idx]) / 2
        tr_y = (self.real_ys[top_idx] + self.real_ys[right_idx]) / 2
        bl_x = (self.real_xs[bottom_idx] + self.real_xs[left_idx]) / 2
        bl_y = (self.real_ys[bottom_idx] + self.real_ys[left_idx]) / 2
        br_x = (self.real_xs[bottom_idx] + self.real_xs[right_idx]) / 2
        br_y = (self.real_ys[bottom_idx] + self.real_ys[right_idx]) / 2
        min_x = max(tl_x, bl_x)
        max_x = min(tr_x, br_x)
        min_y = max(bl_y, br_y)
        max_y = min(tl_y, tr_y)
        sel_x = np.logical_and(self.real_xs >= min_x, self.real_xs <= max_x)
        sel_y = np.logical_and(self.real_ys >= min_y, self.real_ys <= max_y)
        sel = np.logical_and(sel_x, sel_y)
        # import pdb; pdb.set_trace()
        self.ax.scatter(self.real_xs[sel].tolist(),
                        self.real_ys[sel].tolist(),
                        marker='o',
                        alpha=0.5)

    def setReciprocalVectors(self):
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
        with plotutils.get_pyplot() as plt:
            fig = plt.figure(figsize=(10,6))
            self.ax = fig.add_subplot(1, 1, 1)
            self.ax.set_aspect('equal')
            self.plotVect(self.a_pnt, 'a', color='b')
            self.plotVect(self.b_pnt, 'b', color='b')
            self.plotRealGrids()
            self.plotVect(self.ga_vect, 'a*', color='r')
            self.plotVect(self.gb_vect, 'b*', color='r')
            if self.options.interactive:
                print(f"Showing the plot. Click X to close and continue..")
                plt.show(block=True)
            fname = self.options.jobname + self.PNG_EXT
            fig.savefig(fname)
        log(f'Figure saved as {fname}')

    def plotVect(self, pnt, text, color='b'):
        self.ax.arrow(*self.origin, *pnt, linestyle='-', color=color)
        self.ax.annotate("",
                         xy=pnt,
                         xytext=self.origin,
                         arrowprops=dict(arrowstyle="->", color=color))
        self.ax.annotate(text, xy=(pnt + self.origin) / 2, color=color)


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
