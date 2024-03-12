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


class LatticePlotter:

    def __init__(self, ax, a_vect=None, b_vect=None, indices=None, color='b'):
        """
        :param ax 'matplotlib.axes._axes.Axes': axis to plot
        :param a_vect: a vector
        :param b_vect: b vector
        :param indices: the Miller Indexes
        :param color: the color of the lines and arrows
        """
        self.ax = ax
        self.a_vect = a_vect
        self.b_vect = b_vect
        self.indices = indices
        self.color = color
        self.origin = np.array([0., 0.])

    def run(self):
        """
        Main method to run.
        """
        self.plotVect(self.a_vect, 'a', color=self.color)
        self.plotVect(self.b_vect, 'b', color=self.color)
        self.setGrids()
        self.plotGrids()
        self.setVects()
        self.plotVect(self.ma_vect, 'ma', color='r', linestyle="--")
        self.plotVect(self.mb_vect, 'mb', color='r', linestyle="--")
        self.plotPlanes(index=-1)
        self.plotPlanes(index=0)
        self.plotPlanes(index=1)
        self.plotPlanes(index=2)
        self.setPlotStyle()

    def plotVect(self, vect, text, color='b', linestyle="-"):
        """
        Plot an arrow for the vector.

        :param vect list: list of two points
        :param text: the label for the arrow
        :param color: the color of the arrow
        :param linestyle: the line style of the arrow
        """
        if not any(vect):
            return
        arrowprops = dict(linestyle=linestyle, arrowstyle="->", color=color)
        self.ax.annotate("",
                         xy=vect,
                         xytext=self.origin,
                         arrowprops=arrowprops)
        self.ax.annotate(text, xy=(vect + self.origin) / 2, color=color)

    def setGrids(self, num=6):
        """
        Set the grids based on the lattice vectors.

        :param num int:
        """
        xv, yv = np.meshgrid(range(-num, num + 1), range(-num, num + 1))
        self.xs = xv * self.a_vect[0] + yv * self.b_vect[0]
        self.ys = xv * self.a_vect[1] + yv * self.b_vect[1]

    def plotGrids(self):
        """
        Crop the grids by a rectangular and plot.
        """
        bottom_idx = np.unravel_index(self.ys.argmin(), self.ys.shape)
        top_idx = np.unravel_index(self.ys.argmax(), self.ys.shape)
        left_idx = np.unravel_index(self.xs.argmin(), self.xs.shape)
        right_idx = np.unravel_index(self.xs.argmax(), self.xs.shape)
        tl_x = (self.xs[top_idx] + self.xs[left_idx]) / 2
        tl_y = (self.ys[top_idx] + self.ys[left_idx]) / 2
        tr_x = (self.xs[top_idx] + self.xs[right_idx]) / 2
        tr_y = (self.ys[top_idx] + self.ys[right_idx]) / 2
        bl_x = (self.xs[bottom_idx] + self.xs[left_idx]) / 2
        bl_y = (self.ys[bottom_idx] + self.ys[left_idx]) / 2
        br_x = (self.xs[bottom_idx] + self.xs[right_idx]) / 2
        br_y = (self.ys[bottom_idx] + self.ys[right_idx]) / 2
        self.min_x = max(tl_x, bl_x)
        self.max_x = min(tr_x, br_x)
        self.min_y = max(bl_y, br_y)
        self.max_y = min(tl_y, tr_y)
        sel_x = np.logical_and(self.xs >= self.min_x, self.xs <= self.max_x)
        sel_y = np.logical_and(self.ys >= self.min_y, self.ys <= self.max_y)
        sel = np.logical_and(sel_x, sel_y)
        sel_xs = (self.xs[sel] + self.origin[0]).tolist()
        sel_ys = (self.ys[sel] + self.origin[1]).tolist()
        self.ax.scatter(sel_xs, sel_ys, marker='o', alpha=0.5)

    def setVects(self):
        """
        Set the vectors for Miller Plane.
        """
        self.ma_vect = self.a_vect * self.indices[0]
        self.mb_vect = self.b_vect * self.indices[1]

    def plotPlanes(self, index=1):
        """
        Plot the Miller plane moved by the index factor.

        :param index int: by this factor the Miller plane is moved.
        """
        if index == 0:
            minus_pnts = self.getPoints(index=-1)
            plus_pnts = self.getPoints(index=1)
            pnts = np.average([minus_pnts, plus_pnts], axis=0)
        else:
            pnts = self.getPoints(index=index)

        sel_pnts = [
            pnt for pnt in pnts
            if (pnt[0] >= self.min_x) and (pnt[0] <= self.max_x) and (
                pnt[1] >= self.min_y) and (pnt[1] <= self.max_y)
        ]
        sel_pnts = np.array(sel_pnts)

        if np.isclose(*sel_pnts[:, 0]):
            ymin, ymax = sorted(sel_pnts[:, 1])
            self.ax.vlines(np.average(sel_pnts[:, 0]),
                           ymin,
                           ymax,
                           linestyles='--',
                           colors='r')
            return
        self.ax.plot(sel_pnts[:, 0], sel_pnts[:, 1], linestyle='--', color='r')

    def getPoints(self, index=1):
        """
        Get the point to draw the plane with proper translation.

        :param index int: the Miller index plane is moved by this factor.
        :return list: two points
        """
        ma_vect = self.ma_vect
        if not ma_vect.any():
            ma_vect = self.mb_vect + self.a_vect
        mb_vect = self.mb_vect
        if not mb_vect.any():
            mb_vect = self.ma_vect + self.b_vect
        mb_vect = mb_vect * index
        ma_vect = ma_vect * index
        ab = np.linalg.solve([mb_vect, ma_vect], [1, 1])
        x_pnts = []
        if ab[1]:
            x_pnts = [[x, (1 - ab[0] * x) / ab[1]]
                      for x in [self.min_x, self.max_x]]
        y_pnts = []
        if ab[0]:
            y_pnts = [[(1 - ab[1] * y) / ab[0], y]
                      for y in [self.min_y, self.max_y]]
        return x_pnts + y_pnts

    def setPlotStyle(self):
        """
        Set the style of the plot including axis, title and so on.
        """
        self.ax.set_aspect('equal')
        self.ax.set_title('Real Space')


class ReciprocalLatticePlotter(LatticePlotter):

    def run(self):
        """
        Main method to run.
        """
        self.setIndexes()
        self.plotVect(self.a_vect, 'a', color=self.color)
        self.plotVect(self.b_vect, 'b', color=self.color)
        self.setGrids()
        self.plotGrids()
        self.setVects()
        self.plotVect(self.ma_vect, 'ma', color='r', linestyle="--")
        self.plotVect(self.mb_vect, 'mb', color='r', linestyle="--")
        self.setPlotStyle()

    def setIndexes(self):
        self.indices = [1. / x if x else 0 for x in self.indices]

    def setPlotStyle(self):
        super().setPlotStyle()
        self.ax.set_title('Reciprocal Space')


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
        with plotutils.get_pyplot() as plt:
            fig = plt.figure(figsize=(15, 9))

            self.ax1 = fig.add_subplot(1, 2, 1)
            self.ax2 = fig.add_subplot(1, 2, 2)
            LatticePlotter(self.ax1,
                           a_vect=self.a_vect,
                           b_vect=self.b_vect,
                           indices=self.options.miller_indices).run()
            ReciprocalLatticePlotter(
                self.ax2,
                a_vect=self.ga_vect,
                b_vect=self.gb_vect,
                indices=self.options.miller_indices).run()
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
