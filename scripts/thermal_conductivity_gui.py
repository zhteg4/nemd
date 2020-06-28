from PyQt5 import QtCore, QtGui, QtWidgets
import widgets
from pyqtgraph import PlotWidget, plot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import sys  # We need sys so that we can pass argv to QApplication
import os


class Canvas(FigureCanvasQTAgg):
    def __init__(self, nsubplot=111, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(nsubplot)
        super().__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.central_layout = QtGui.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(self.central_layout)
        self.setCentralWidget(central_widget)

        self.canvas = Canvas()
        self.central_layout.addWidget(self.canvas)

        self.thermal_conductivity_le = widgets.LineEdit(
            '',
            label='Thermal Conductivity:',
            after_label='W/(m⋅K)',
            layout=self.central_layout,
            readonly=True)

        hour = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        temperature = [30, 32, 34, 32, 33, 31, 29, 32, 35, 45]
        self.canvas.axes.plot(hour, temperature)
        # self.statusBar().showMessage('Ready')

        self.setWindowTitle('Thermal Conductivity Viewer')


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
