# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.

# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module provides backend plotting for drivers.
"""
import sh
import numpy as np
import pandas as pd
import adjustText
from nemd import constants
from nemd import environutils
from contextlib import contextmanager


@contextmanager
def get_pyplot(inav=None, name='the plot'):
    """
    Get the pyplot with requested backend and restoration after usage.

    :param inav: use Agg on False (no show)
    :type inav: bool
    :param name str: the name of the plot
    :return: the pyplot with requested backend
    :rtype: module 'matplotlib.pyplot'
    """
    if inav is None:
        inav = environutils.is_interactive()
    import matplotlib
    obackend = matplotlib.get_backend()
    backend = obackend if inav else 'Agg'
    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    try:
        yield plt
    finally:
        if inav:
            print(f"Showing {name}. Click X to close and continue..")
            plt.show(block=True)
        plt.close('all')
        # Restore the backend
        matplotlib.use(obackend)


class TempEnePlotter(object):

    def __init__(
        self,
        lammp_temp,
        lammps_energy,
        jobname,
    ):
        self.lammp_temp = lammp_temp
        self.lammps_energy = lammps_energy
        self.jobname = jobname
        self.interactive = environutils.is_interactive()
        self.fig_file = self.jobname + '.png'
        self.fig_nrows = 2
        self.fig_ncols = 1

    def load(self):
        self.temp_data = self.lammp_temp.data
        self.ene_data = self.lammps_energy.data
        self.fitted_temp_data = self.lammp_temp.fitted_data
        self.fitted_ene_data = self.lammps_energy.fitted_data
        self.temp_data_nrow, self.temp_data_ncol, self.temp_data_nblock = self.temp_data.shape
        self.ene_names = self.ene_data.dtype.names

    def setFigure(self):
        if self.interactive:
            return

        import matplotlib
        self.old_backed = matplotlib.pyplot.get_backend()
        matplotlib.use("agg", force=True)
        from matplotlib import pyplot as plt

        self.fig = plt.figure()
        self.temp_axis = self.fig.add_subplot(self.fig_nrows, self.fig_ncols,
                                              1)
        self.ene_axis = self.fig.add_subplot(self.fig_nrows, self.fig_ncols, 2)

    def plot(self):
        self.load()
        self.setFigure()
        self.plotTemp()
        self.plotEne()
        self.setLayout()
        self.show()
        self.save()
        self.resetMatplotlib()

    def save(self):
        self.fig.savefig(self.fig_file)

    def resetMatplotlib(self):

        if self.interactive:
            return

        import matplotlib
        matplotlib.use(self.old_backed)

    def plotEne(self):
        self.ene_axis.plot(self.ene_data[self.ene_names[0]],
                           -self.ene_data[self.ene_names[2]],
                           label=self.ene_names[2])
        self.ene_axis.plot(self.ene_data[self.ene_names[0]],
                           self.ene_data[self.ene_names[3]],
                           label=self.ene_names[3])
        if self.fitted_ene_data is not None:
            self.ene_axis.plot(self.fitted_ene_data[:, 0],
                               self.fitted_ene_data[:, 1],
                               label='Fitted')
        self.ene_axis.set_xlabel(self.ene_names[0])
        self.ene_axis.set_ylabel(f'Energy {self.ene_names[3].split()[-1]}')
        self.ene_axis.legend(loc='upper left', prop={'size': 6})

    def plotTemp(self):

        for iblock in range(self.temp_data_nblock - 1):
            self.temp_axis.plot(self.temp_data[:, 1, iblock],
                                self.temp_data[:, 3, iblock],
                                '.',
                                label=f'Block {iblock}')
        self.temp_axis.plot(self.temp_data[:, 1, -1],
                            self.temp_data[:, 3, -1],
                            label='Average')
        if self.fitted_temp_data is not None:
            self.temp_axis.plot(self.fitted_temp_data[:, 0],
                                self.fitted_temp_data[:, 1],
                                label='Fitted')
        self.temp_axis.legend(loc='upper right', prop={'size': 6})
        self.temp_axis.set_ylim([270, 330])
        self.temp_axis.set_xlabel('Coordinate (Angstrom)')
        self.temp_axis.set_ylabel('Temperature (K)')

    def setLayout(self):
        self.fig.tight_layout()

    def show(self):
        if not self.interactive:
            return

        self.fig.show()
        input(
            'Showing the temperature profile and energy plots. Press any keys to continue...'
        )


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
        self.vect = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.origin = np.array([0., 0.])
        self.texts = []

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
        return np.array(sel_pnts)[:2, :]

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

    def setPlotStyle(self):
        """
        Set the style of the plot including axis, title and so on.
        """
        super().setPlotStyle()
        self.ax.set_title('Reciprocal Space')


class DispersionPlotter:

    THZ = 'THz'

    def __init__(self, filename, unit=THZ):
        """
        :param filename str: the file containing the dispersion data
        :param unit str: the unit of the y data (either THz or cm^-1)
        """
        self.filename = filename
        self.unit = unit
        self.data = None
        self.ymin, self.ymax = None, None
        self.fig = None

    def run(self):
        """
        Main method to run.
        """
        self.readData()
        self.setKpoints()
        self.setFigure()

    def readData(self):
        """
        Read the data from the file with unit conversion and range set.
        """
        data = pd.read_csv(self.filename, header=None, skiprows=3, sep=b'\s+')
        self.data = data.set_index(0)
        if self.unit == self.THZ:
            self.data *= constants.CM_INV_THZ
        self.ymin = min([0, self.data.min().min()])
        self.ymax = self.data.max().max() * 1.05

    def setKpoints(self):
        """
        Set the point values and labels.
        """
        header = sh.head('-n', '2', self.filename).split('\n')[:2]
        symbols, pnts = [x.strip('#').split() for x in header]
        pnts = [float(x) for x in pnts]
        # Adjacent K points may have the same value
        same_ids = [i for i in range(1, len(pnts)) if pnts[i - 1] == pnts[i]]
        idxs = [x for x in range(len(pnts)) if x not in same_ids]
        self.pnts = [pnts[i] for i in idxs]
        self.symbols = [symbols[i] for i in idxs]
        for id in same_ids:
            # Adjacent K points with the same value combine the labels
            self.symbols[id - 1] = '|'.join([symbols[id - 1], symbols[id]])

    def setFigure(self):
        """
        Plot the frequency vs wave vector with k-point vertical lines.
        """
        with get_pyplot() as plt:
            self.fig = plt.figure(figsize=(10, 6))
            ax = self.fig.add_subplot(1, 1, 1)
            for column in self.data.columns:
                ax.plot(self.data.index,
                        self.data[column],
                        linestyle='-',
                        color='b')
            ax.vlines(self.pnts[1:-1],
                      self.ymin,
                      self.ymax,
                      linestyles='--',
                      color='k')
            ax.set_xlim([self.data.index.min(), self.data.index.max()])
            ax.set_ylim([self.ymin, self.ymax])
            ax.set_xticks(self.pnts)
            ax.set_xticklabels(self.symbols)
            ax.set_xlabel('Wave vector')
            ax.set_ylabel(f'Frequency ({self.unit})')
            self.fig.tight_layout()
