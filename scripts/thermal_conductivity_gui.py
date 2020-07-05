import os
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
from types import SimpleNamespace

import widgets
import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import lines
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class DraggableLine:
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'

    def __init__(self, axis, position=0.1, kind=VERTICAL):
        self.axis = axis
        self.canvas = self.axis.get_figure().canvas
        self.kind = kind
        self.position = position

        x = [position, position]
        y = [-1, 1]
        if kind == self.HORIZONTAL:
            x, y = y, x
        self.line = lines.Line2D(x, y, picker=5)
        self.axis.add_line(self.line)
        self.canvas.draw_idle()
        self.sid = self.canvas.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            self.follower = self.canvas.mpl_connect("motion_notify_event",
                                                    self.followmouse)
            self.releaser = self.canvas.mpl_connect("button_press_event",
                                                    self.releaseonclick)

    def followmouse(self, event, other_demension=None):
        if self.kind == self.HORIZONTAL:
            self.line.set_ydata([event.ydata, event.ydata])
            if other_demension:
                self.line.set_xdata(
                    [other_demension.min_data, other_demension.max_data])
        else:
            self.line.set_xdata([event.xdata, event.xdata])
            if other_demension:
                self.line.set_ydata(
                    [other_demension.min_data, other_demension.max_data])
        if other_demension:
            return
        self.canvas.draw_idle()

    def releaseonclick(self, event):
        if self.kind == self.HORIZONTAL:
            self.position = self.line.get_ydata()[0]
        else:
            self.position = self.line.get_xdata()[0]

        self.canvas.mpl_disconnect(self.releaser)
        self.canvas.mpl_disconnect(self.follower)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.temp_axis = self.fig.add_subplot(211)
        self.ene_axis = self.fig.add_subplot(212)
        super().__init__(self.fig)
        self.temp_lline = DraggableLine(self.temp_axis)
        self.temp_uline = DraggableLine(self.temp_axis, position=0.9)

    def plot(self, temp_data, ene_data):

        coordinates = temp_data[:, 0]
        temp = temp_data[:, 1]
        self.x = coordinates
        self.y = temp
        x_min = min(self.x)
        x_span = max(self.x) - x_min
        y_min = min(self.y)
        y_span = max(self.y) - y_min

        self.temp_lline.followmouse(
            SimpleNamespace(xdata=0.1 * x_span + x_min),
            other_demension=SimpleNamespace(min_data=-0.1 * y_span + y_min,
                                            max_data=1.1 * y_span + y_min))
        self.temp_uline.followmouse(
            SimpleNamespace(xdata=0.9 * x_span + x_min),
            other_demension=SimpleNamespace(min_data=-0.1 * y_span + y_min,
                                            max_data=1.1 * y_span + y_min))

        temp_lower_bound = temp - temp_data[:, 2]
        temp_upper_bound = temp + temp_data[:, 2]
        self.temp_axis.plot(coordinates, temp)
        self.temp_axis.fill_between(coordinates,
                                    temp_lower_bound,
                                    temp_upper_bound,
                                    alpha=0.2,
                                    picker=5)
        self.temp_axis.set_xlim(-0.1 * x_span + x_min, 1.1 * x_span + x_min)
        self.temp_axis.set_ylim(-0.1 * y_span + y_min, 1.1 * y_span + y_min)

        time = ene_data[:, 0]
        energy = ene_data[:, 1]
        energy_lower_bound = energy - ene_data[:, 2]
        energy_upper_bound = energy + ene_data[:, 2]
        self.ene_axis.plot(time, energy)
        self.ene_axis.fill_between(time,
                                   energy_lower_bound,
                                   energy_upper_bound,
                                   alpha=0.2)

        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app, *args, **kwargs):
        self.app = app
        self.file_path = None
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Thermal Conductivity Viewer')
        self.central_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(self.central_layout)
        self.setCentralWidget(central_widget)

        self.load_data_bn = widgets.PushButton('Load Data',
                                               after_label='not set',
                                               layout=self.central_layout,
                                               command=self.loadAndDraw)

        self.canvas = Canvas()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.central_layout.addWidget(self.toolbar)
        self.central_layout.addWidget(self.canvas)
        # matplotlib.widgets.Cursor(self.canvas.temp_axis, vertOn=True)
        # self.slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        # self.central_layout.addWidget(self.slider)
        # self.slider.setMinimum(10)
        # self.slider.setMaximum(30)
        # self.slider.setValue(20)

        self.thermal_conductivity_le = widgets.LineEdit(
            '',
            label='Thermal Conductivity:',
            after_label='W/(m⋅K)',
            layout=self.central_layout,
            readonly=True)

        # hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
        # self.canvas.axes.plot(hour, temperature)
        # self.statusBar().showMessage('Ready')

    def loadAndDraw(self, file_path=None):
        self.setLogFilePath(file_path=file_path)
        self.setLoadDataLabels()
        self.loadData()
        self.draw()

    def setLogFilePath(self, file_path=None):
        if not file_path:
            dlg = QtWidgets.QFileDialog(self)
            dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
            dlg.setNameFilters(["Driver log (*-driver.log)"])
            if dlg.exec_():
                file_path = dlg.selectedFiles()[0]

        if not file_path:
            return

        self.file_path = file_path

    def setLoadDataLabels(self):
        if not self.file_path:
            return

        self.load_data_bn.after_label.setText(os.path.basename(self.file_path))
        self.load_data_bn.after_label.setToolTip(self.file_path)

    def loadData(self):
        if not self.file_path:
            return

        temp_file = self.file_path.replace('-driver.log', '_temp.npz')
        ene_file = self.file_path.replace('-driver.log', '_ene.npz')
        try:
            self.temp_data = np.load(temp_file)['data']
        except FileNotFoundError:
            self.reset()

        try:
            self.ene_data = np.load(ene_file)['data']
        except FileNotFoundError:
            self.reset()

    def draw(self):
        if self.file_path is None:
            return

        self.canvas.plot(self.temp_data, self.ene_data)

    def panel(self):
        self.show()
        sys.exit(self.app.exec_())


def get_panel():
    app = QtWidgets.QApplication(sys.argv)
    panel = MainWindow(app)
    return panel


def main():
    panel = get_panel()
    panel.panel()


if __name__ == '__main__':
    main()
