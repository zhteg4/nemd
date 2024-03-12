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
import adjustText

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
        self.ma_vect = None
        self.mb_vect = None
        self.origin = np.array([0., 0.])
        self.texts = []
        self.vect = None

    def run(self):
        """
        Main method to run.
        """
        self.setGrids()
        self.plotGrids()
        self.plotVect(self.a_vect, 'a', color=self.color)
        self.plotVect(self.b_vect, 'b', color=self.color)
        self.setVects()
        self.plotVect(self.ma_vect, 'ma', color='r', linestyle="--")
        self.plotVect(self.mb_vect, 'mb', color='r', linestyle="--")
        self.plotPlanes(index=-1)
        self.plotPlanes(index=0)
        self.plotPlanes(index=1)
        self.plotPlaneNorm()
        self.setPlotStyle()

    def plotVect(self,
                 vect,
                 text,
                 xytext=None,
                 color='b',
                 linestyle="-",
                 arrowstyle="->"):
        """
        Plot an arrow for the vector.

        :param vect list: list of two points
        :param text: the label for the arrow
        :param color: the color of the arrow
        :param linestyle: the line style of the arrow
        """
        if not any(vect):
            return
        if xytext is None:
            xytext = self.origin
        arrowprops = dict(linestyle=linestyle,
                          arrowstyle=arrowstyle,
                          color=color)
        self.ax.annotate("", xy=vect, xytext=xytext, arrowprops=arrowprops)
        text = self.ax.annotate(text, xy=(vect + xytext) / 2, color=color)
        self.texts.append(text)

    def setGrids(self, num=6):
        """
        Set the grids based on the lattice vectors, and crop the grids by a
        rectangular.

        :param num int: the minimum number of duplicates along each lattice vector.
        """
        num = max(self.indices + [num]) + 2
        xv, yv = np.meshgrid(range(-num, num + 1), range(-num, num + 1))
        xs = xv * self.a_vect[0] + yv * self.b_vect[0]
        ys = xv * self.a_vect[1] + yv * self.b_vect[1]
        bottom_idx = np.unravel_index(ys.argmin(), ys.shape)
        top_idx = np.unravel_index(ys.argmax(), ys.shape)
        left_idx = np.unravel_index(xs.argmin(), xs.shape)
        right_idx = np.unravel_index(xs.argmax(), xs.shape)
        tl_x = (xs[top_idx] + xs[left_idx]) / 2
        tl_y = (ys[top_idx] + ys[left_idx]) / 2
        tr_x = (xs[top_idx] + xs[right_idx]) / 2
        tr_y = (ys[top_idx] + ys[right_idx]) / 2
        bl_x = (xs[bottom_idx] + xs[left_idx]) / 2
        bl_y = (ys[bottom_idx] + ys[left_idx]) / 2
        br_x = (xs[bottom_idx] + xs[right_idx]) / 2
        br_y = (ys[bottom_idx] + ys[right_idx]) / 2
        self.min_x = max(tl_x, bl_x)
        self.max_x = min(tr_x, br_x)
        self.min_y = max(bl_y, br_y)
        self.max_y = min(tl_y, tr_y)
        sel_x = np.logical_and(xs >= self.min_x, xs <= self.max_x)
        sel_y = np.logical_and(ys >= self.min_y, ys <= self.max_y)
        sel = np.logical_and(sel_x, sel_y)
        self.xs = (xs[sel] + self.origin[0]).tolist()
        self.ys = (ys[sel] + self.origin[1]).tolist()

    def plotGrids(self):
        """
        Plot the cropped grids.
        """
        self.ax.scatter(self.xs, self.ys, marker='o', alpha=0.5)

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

        sel_pnts = self.selectPoints(index=index)
        if np.isclose(*sel_pnts[:, 0]):
            ymin, ymax = sorted(sel_pnts[:, 1])
            self.ax.vlines(np.average(sel_pnts[:, 0]),
                           ymin,
                           ymax,
                           linestyles='--',
                           colors='r')
            return
        self.ax.plot(sel_pnts[:, 0], sel_pnts[:, 1], linestyle='--', color='r')

    def selectPoints(self, index=1):
        """
        Get two intersection points of the miller plane crossing the boundary.

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
        return np.array(sel_pnts)

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

    def plotPlaneNorm(self):
        """
        Plot the normal to a plane.
        """
        pnt1 = self.getDPoint(index=0)
        pnt2 = self.getDPoint(index=1)
        self.vect = pnt2 - pnt1
        norm = np.linalg.norm(self.vect)
        self.plotVect(pnt2,
                      f'd={norm:.4g}',
                      xytext=pnt1,
                      color='g',
                      arrowstyle='<->')
        log(f"The vector in reciprocal Space is: {self.vect} with {norm} as the norm."
            )

    def getDPoint(self, index=1):
        """
        Get the intersection between the plane and its normal.

        :param index int: the Miller index to define the plane.
        :return list of two float: the intersection point
        """
        pnts = self.selectPoints(index=index)
        vect = pnts[1] - pnts[0]
        d_vect = np.dot([[0, 1], [-1, 0]], vect)
        factor = np.linalg.solve(np.transpose([d_vect, -vect]), pnts[0])[0]
        return d_vect * factor

    def setPlotStyle(self):
        """
        Set the style of the plot including axis, title and so on.
        """
        adjustText.adjust_text(self.texts, ax=self.ax)
        self.ax.set_aspect('equal')
        self.ax.set_title('Real Space')


class ReciprocalLatticePlotter(LatticePlotter):

    def run(self):
        """
        Main method to run.
        """
        self.setGrids()
        self.plotGrids()
        self.setIndexes()
        self.plotVect(self.a_vect, 'a', color=self.color)
        self.plotVect(self.b_vect, 'b', color=self.color)
        self.setVects()
        self.plotVect(self.ma_vect, 'ma', color='r', linestyle="--")
        self.plotVect(self.mb_vect, 'mb', color='r', linestyle="--")
        self.plotVectSummation()
        self.setPlotStyle()

    def setIndexes(self):
        """
        Set the reciprocal index (lattice scalar) from real space Miller index.
        """
        self.indices = [1. / x if x else 0 for x in self.indices]

    def plotVectSummation(self):
        """
        Plot the vector summation.
        """
        self.vect = self.ma_vect + self.mb_vect
        norm = np.linalg.norm(self.vect)
        self.plotVect(self.vect, f'r={norm:.4g}', color='g', linestyle="--")
        log(f"The vector in reciprocal Space is: {self.vect} with {norm} being "
            "the norm.")

    def setPlotStyle(self):
        """
        Set the style of the plot including axis, title and so on.
        """
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
        """
        Plot the real and reciprocal paces.
        """
        with plotutils.get_pyplot() as plt:
            fig = plt.figure(figsize=(15, 9))

            self.ax1 = fig.add_subplot(1, 2, 1)
            self.ax2 = fig.add_subplot(1, 2, 2)
            ltp = LatticePlotter(self.ax1,
                                 a_vect=self.a_vect,
                                 b_vect=self.b_vect,
                                 indices=self.options.miller_indices)
            ltp.run()
            rltp = ReciprocalLatticePlotter(
                self.ax2,
                a_vect=self.ga_vect,
                b_vect=self.gb_vect,
                indices=self.options.miller_indices)
            rltp.run()
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
