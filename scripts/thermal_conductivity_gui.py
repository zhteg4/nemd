from PyQt5 import QtCore, QtGui, QtWidgets
import widgets
from pyqtgraph import PlotWidget, plot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
import sys  # We need sys so that we can pass argv to QApplication
import os
import numpy as np


class Canvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.temp_axis = self.fig.add_subplot(211)
        self.ene_axis = self.fig.add_subplot(212)
        super().__init__(self.fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, app, *args, **kwargs):
        self.app = app
        self.file_path = None
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Thermal Conductivity Viewer')
        self.central_layout = QtGui.QVBoxLayout()
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
        self.load_data_bn.after_label.setText(os.path.basename(self.file_path))
        self.load_data_bn.after_label.setToolTip(self.file_path)

    def loadData(self):
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
        if self.temp_data is None:
            return

        coordinates = self.temp_data[:, 0]
        temp = self.temp_data[:, 1]
        temp_lower_bound = temp - self.temp_data[:, 2]
        temp_upper_bound = temp + self.temp_data[:, 2]
        self.canvas.temp_axis.plot(self.temp_data[:, 0], self.temp_data[:, 1])
        self.canvas.temp_axis.fill_between(coordinates,
                                           temp_lower_bound,
                                           temp_upper_bound,
                                           alpha=0.2)
        time = self.ene_data[:, 0]
        energy = self.ene_data[:, 1]
        energy_lower_bound = energy - self.ene_data[:, 2]
        energy_upper_bound = energy + self.ene_data[:, 2]
        self.canvas.ene_axis.plot(time, energy)
        self.canvas.ene_axis.fill_between(time,
                                          energy_lower_bound,
                                          energy_upper_bound,
                                          alpha=0.2)
        self.canvas.draw()

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
