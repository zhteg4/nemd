import os
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from nemd import symbols
from nemd import plotutils


class Press:

    DATA = 'Data'
    PRESS = 'press'
    VOL = 'vol'
    PNG_EXT = '.png'

    def __init__(self, filename):
        """
        :param filename str: the filename with path to load data from
        """
        self.filename = filename
        self.data = None
        self.ave_press = None

    def run(self):
        """
        Main method to run.
        """
        self.setData()
        self.setAve()
        self.plot()

    def setData(self):
        """
        Load data from the file.
        """
        self.data = pd.read_csv(self.filename,
                                sep=r'\s+',
                                header=1,
                                na_filter=False,
                                escapechar='#',
                                index_col=0)

    def setAve(self):
        """
        Set the averaged data.
        """
        press_lb = self.getColumn(self.PRESS)
        self.ave_press = self.data[press_lb].mean()

    def getColumn(self, ending=PRESS):
        """
        Get the column label based the ending str.

        :param ending str: select the label ends with this string.
        :return str: selected column label
        """
        return [x for x in self.data.columns if x.endswith(ending)][0]

    @staticmethod
    def getLabel(column):
        """
        Shape the label for visualization.

        :param column str: one data column label
        :return str: label to be displayed on the figure.
        """
        column = column.removeprefix('c_').removeprefix('v_').split('_')
        return ' '.join([x.capitalize() for x in column])

    def plot(self):
        """
        To be overwritten.
        """
        pass


class BoxLength(Press):

    XL = 'xl'
    YL = 'yl'
    ZL = 'zl'

    def __init__(self, filename, last_pct=0.2, ending=XL):
        """
        :param filename str: the filename with path to load data from
        :param last_pct float: the last this percentage of the data are used
        :param ending str: the data column ending with str is used.
        """
        super().__init__(filename)
        self.last_pct = last_pct
        self.sel_length = None
        self.sindex = None
        self.ending = ending

    def setAve(self):
        """
        Get the box length in one dimension.

        :return float: the averaged box length in one dimension.
        """
        column = self.getColumn(self.ending)
        data = self.data[column]
        self.sindex = math.floor(data.shape[0] * (1 - self.last_pct))
        self.ave_length = data[self.sindex:].mean()

    def plot(self):
        """
        To be overwritten.
        """
        with plotutils.get_pyplot(inav=False) as plt:
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
            column = self.getColumn(self.ending)
            col = self.data[column]
            ax.plot(self.data.index, col, label=self.DATA)
            ax.plot(self.data.index[self.sindex:],
                    col[self.sindex:],
                    'g',
                    label='Selected')
            ax.set_xlabel(self.data.index.name)
            ax.set_ylabel(self.getLabel(column))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels)
            basename = os.path.basename(self.filename)
            name = symbols.PERIOD.join(basename.split(symbols.PERIOD)[:-1])
            fig.savefig(f"{name}_{self.ending}{self.PNG_EXT}")


class Modulus(Press):

    MODULUS = 'modulus'
    DEFAULT = 10
    STD_DEV = '_(Std_Dev)'
    SMOOTHED = '_(Smoothed)'

    def __init__(self, filename, record_num):
        """
        :param filename str: the filename with path to load data from
        :param record_num int: the recording number of each cycle.
        """
        super().__init__(filename)
        self.record_num = record_num
        self.ave = pd.DataFrame()
        self.modulus = None

    def run(self):
        """
        Main method to run.
        """
        super().run()
        self.setModulus()

    def setAve(self):
        """
        Set the averaged data.
        """
        for column in self.data.columns:
            col = self.data[column].values
            mod = col.shape[0] % self.record_num
            if mod:
                col = np.concatenate(([np.nan], col))
            data = col.reshape(-1, self.record_num)
            self.ave[column] = np.nanmean(data, axis=0)
            self.ave[column + self.STD_DEV] = np.nanstd(data, axis=0)
            smoothed_lb = column + self.SMOOTHED
            window = int(self.record_num / 10)
            self.ave[smoothed_lb] = savgol_filter(self.ave[column], window, 3)

    def plot(self):
        """
        Plot the data and save the figure.
        """
        with plotutils.get_pyplot(inav=False) as plt:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
            for id, (axis, column) in enumerate(zip(axes, self.data.columns)):
                self.subplot(axis, column)
                if not id:
                    num = round(self.data.shape[0] / self.record_num)
                    axis.set_title(f"Sinusoidal Deformation ({num} cycles)")
            basename = os.path.basename(self.filename)
            name = symbols.PERIOD.join(basename.split(symbols.PERIOD)[:-1])
            fig.savefig(f"{name}_{self.MODULUS}{self.PNG_EXT}")

    def subplot(self, ax, column):
        """
        Plot the data corresponding to column label on the axis.

        :param ax 'matplotlib.axes._axes.Axes':the axis to plot
        :param column str: the column of the data
        """
        ax.plot(self.ave.index, self.ave[column], label=self.DATA)
        smoothed_lb = column + self.SMOOTHED
        ax.plot(self.ave.index, self.ave[smoothed_lb], label="Smoothed")
        std_dev = self.ave[column + self.STD_DEV]
        lbndry = self.ave[column] - std_dev
        ubndry = self.ave[column] + std_dev
        ax.fill_between(self.ave.index, lbndry, ubndry, alpha=0.5, label="SD")
        ax.set_xlabel(self.data.index.name)
        ax.set_ylabel(self.getLabel(column))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)

    def setModulus(self):
        """
        Set the bulk modulus.

        :return float: the bulk modulus from cycles.
        """
        press_lb = self.getColumn(self.PRESS) + self.SMOOTHED
        press_delta = self.ave[press_lb].max() - self.ave[press_lb].min()
        vol_lb = self.getColumn(self.VOL) + self.SMOOTHED
        vol_delta = self.ave[vol_lb].max() - self.ave[vol_lb].min()
        modulus = press_delta / vol_delta * self.ave[vol_lb].mean()
        self.modulus = max([modulus, self.DEFAULT])


class Scale(Press):

    SCALE = 'scale'
    FITTED = 'Fitted'

    def __init__(self, press, *args, **kwargs):
        """
        :param filename str: the filename with path to load data from
        :param record_num int: the recording number of each cycle.
        """
        self.press = press
        super().__init__(*args, **kwargs)
        self.vol = None
        self.factor = None

    def run(self):
        """
        Main method to run.
        """
        super().run()
        self.setFactor()

    def plot(self):
        """
        Plot the data and save the figure.
        """
        with plotutils.get_pyplot(inav=False) as plt:
            fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))
            labels = {x: self.getLabel(x) for x in self.data.columns}
            press_cl = [x for x in labels if x.endswith(self.PRESS)][0]
            vol_cl = [x for x in labels if x.endswith(self.VOL)][0]
            ax.plot(self.data[vol_cl],
                    self.data[press_cl],
                    '.',
                    label=self.DATA)
            coef = np.polyfit(self.data[vol_cl], self.data[press_cl], 1)
            vmin, vmax = self.data[vol_cl].min(), self.data[vol_cl].max()
            self.vol = np.linspace(vmin, vmax, 100)
            self.fitted_press = np.poly1d(coef)(self.vol)
            ax.plot(self.vol, self.fitted_press, '--', label=self.FITTED)
            ax.set_title(
                f"{self.VOL.capitalize()} vs {self.PRESS.capitalize()}")
            basename = os.path.basename(self.filename)
            name = symbols.PERIOD.join(basename.split(symbols.PERIOD)[:-1])
            fig.savefig(f"{name}_{self.SCALE}{self.PNG_EXT}")

    def setFactor(self):
        """
        Set the scale factor.
        """
        index = abs(self.fitted_press - self.press).argmin()
        vol = self.vol[index]
        delta = (self.vol[-1] - self.vol[0]) * 0.05
        left, right = vol - delta, vol + delta
        sel_ids = (self.vol > left) & (self.vol < right)
        ratio = self.vol[sel_ids].shape[0] / self.vol.shape[0]
        self.factor = 1 if ratio > 0.06 else vol / self.vol.mean()


def getPress(filename):
    """
    Get the averaged pressure.

    :param filename str: the filename with path to load data from
    :return float: averaged pressure.
    """
    press = Press(filename)
    press.run()
    return press.ave_press


def getModulus(filename, record_num):
    """
    Get the bulk modulus.

    :param filename str: the filename with path to load data from
    :param record_num int: the recording number of each cycle.
    :return float: the bulk modulus.
    """
    modulus = Modulus(filename, record_num)
    modulus.run()
    return modulus.modulus


def getScaleFactor(press, filename):
    """
    Get the scale factor of the volume. The scale factor is the ratio of the
    volume at the closet target pressure to the volume at the average pressure.

    :param press float: the target pressure.
    :param filename str: the filename with path to load data from.
    :return float: the scale factor of the volume.
    """
    scale = Scale(press, filename)
    scale.run()
    return scale.factor


def getBdryFactor(press, filename):
    """
    Get the scale factor of the volume. The scale factor is the ratio of the
    volume at the closet target pressure to the volume at the average pressure.

    :param press float: the target pressure.
    :param filename str: the filename with path to load data from.
    :return float: the scale factor of the volume.
    """
    scale = Scale(press, filename)
    scale.run()
    return getScaleFactor(press, filename)**(1 / 3)


def getXL(filename):
    """
    Get the box length in the x dimension.

    :param filename str: the filename with path to load data from
    :return float: box length
    """
    return getL(filename, ending=BoxLength.XL)


def getYL(filename):
    """
    Get the box length in the y dimension.

    :param filename str: the filename with path to load data from
    :return float: box length
    """
    return getL(filename, ending=BoxLength.YL)


def getZL(filename):
    """
    Get the box length in the z dimension.

    :param filename str: the filename with path to load data from
    :return float: box length
    """
    return getL(filename, ending=BoxLength.ZL)


def getL(filename, last_pct=0.2, ending=BoxLength.XL):
    """
    Get the box length in the one dimension.

    :param filename str: the filename with path to load data from
    :param last_pct float: the last this percentage of the data are used
    :param ending str: select the label ends with this string
    :return float: box length
    """
    box_length = BoxLength(filename, last_pct=last_pct, ending=ending)
    box_length.run()
    return box_length.ave_length
