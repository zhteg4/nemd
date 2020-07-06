import os
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
from types import SimpleNamespace

import nemd
import widgets
import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import lines
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class DraggableLine(object):

    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'

    def __init__(self, axis, position=0.1, kind=VERTICAL):
        self.axis = axis
        self.canvas = self.axis.get_figure().canvas
        self.kind = kind
        self.position = position

        x_line = [position, position]
        y_line = [-1, 1]
        if kind == self.HORIZONTAL:
            x_line, y_line = y_line, x_line
        self.line = lines.Line2D(x_line,
                                 y_line,
                                 color='m',
                                 linewidth=2.5,
                                 alpha=0.3,
                                 picker=5)
        self.axis.add_line(self.line)
        self.canvas.draw_idle()
        self.sid = self.canvas.mpl_connect('pick_event', self.clickonline)

    def clickonline(self, event):
        if event.artist == self.line:
            self.follower = self.canvas.mpl_connect("motion_notify_event",
                                                    self.followmouse)
            self.releaser = self.canvas.mpl_connect("button_press_event",
                                                    self.releaseonclick)

    def followmouse(self, event):
        self.setLinePosition(event)
        self.canvas.draw_idle()

    def update(self, event):
        self.setLinePosition(event)
        self.setLineLength()

    def setLineLength(self):
        if self.kind == self.VERTICAL:
            y_min, y_max = self.axis.get_ylim()
            self.line.set_ydata([y_min, y_max])
        else:
            x_min, x_max = self.axis.get_xlim()
            self.line.set_xdata([x_min, x_max])

    def setLinePosition(self, event):
        if self.kind == self.VERTICAL:
            x_point = event.xdata
            self.line.set_xdata([x_point, x_point])
        else:
            y_point = event.ydata
            self.line.set_ydata([y_point, y_point])

    def releaseonclick(self, event):
        self.setPosition()
        self.canvas.mpl_disconnect(self.releaser)
        self.canvas.mpl_disconnect(self.follower)

    def setPosition(self):
        if self.kind == self.HORIZONTAL:
            self.position = self.line.get_ydata()[0]
        else:
            self.position = self.line.get_xdata()[0]


class LineWithVSpan(DraggableLine):

    LEFT = 'left'
    RIGHT = 'right'

    def __init__(self, *args, fill_direction='right', **kwargs):
        self.fill_direction = fill_direction
        super().__init__(*args, **kwargs)
        xrange = list(self.axis.get_xlim())
        edge_index = 0 if fill_direction == self.RIGHT else 1
        xrange[edge_index] = self.position
        self.polygon = self.axis.axvspan(*xrange, alpha=0.3, color='grey')

    def followmouse(self, event):
        self.setLinePosition(event)
        self.resizeVSpan(event)
        self.canvas.draw_idle()

    def update(self, event):
        self.resizeVSpan(event)
        super().update(event)
        self.setPosition()

    def resizeVSpan(self, event):
        vpan_xy = self.polygon.get_xy()
        xmin, xmax = self.axis.get_xlim()
        x_lim = xmax if self.fill_direction == self.RIGHT else xmin
        print(vpan_xy[:, 0])
        vpan_xy[:, 0] = [event.xdata, event.xdata, x_lim, x_lim, event.xdata]
        self.polygon.set_xy(vpan_xy)


class Canvas(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.temp_axis = self.fig.add_subplot(211)
        self.ene_axis = self.fig.add_subplot(212)
        super().__init__(self.fig)
        self.temp_axis.set_xlim((0, 1))
        self.temp_lline = LineWithVSpan(self.temp_axis,
                                        fill_direction=LineWithVSpan.LEFT)
        self.temp_rline = LineWithVSpan(self.temp_axis, position=0.9)

    def plot(self,
             temp_data,
             ene_data,
             lim_frac=(-0.1, 1.1),
             line_frac=(0.1, 0.9)):

        coordinates = temp_data[:, 0]
        temp = temp_data[:, 1]
        coord_min = min(coordinates)
        coord_span = max(coordinates) - coord_min
        temp_min = min(temp)
        temp_span = max(temp) - temp_min

        self.temp_axis.set_xlim(lim_frac[0] * coord_span + coord_min,
                                lim_frac[1] * coord_span + coord_min)
        self.temp_axis.set_ylim(lim_frac[0] * temp_span + temp_min,
                                lim_frac[1] * temp_span + temp_min)

        self.temp_lline.update(
            SimpleNamespace(xdata=line_frac[0] * coord_span + coord_min))
        self.temp_rline.update(
            SimpleNamespace(xdata=line_frac[1] * coord_span + coord_min))

        temp_lower_bound = temp - temp_data[:, 2]
        temp_upper_bound = temp + temp_data[:, 2]
        self.temp_axis.plot(coordinates, temp)
        self.temp_axis.fill_between(coordinates,
                                    temp_lower_bound,
                                    temp_upper_bound,
                                    alpha=0.2,
                                    picker=5)

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


class NemdPanel(QtWidgets.QMainWindow):
    def __init__(self, app, *args, **kwargs):
        self.app = app
        self.file_path = None
        super(NemdPanel, self).__init__(*args, **kwargs)
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

        hlayout = QtWidgets.QHBoxLayout()
        self.central_layout.addLayout(hlayout)
        self.thermal_conductivity_le = widgets.FloatLineEdit(
            '',
            label='Thermal Conductivity:',
            after_label='W/(m⋅K)',
            layout=hlayout,
            readonly=True)
        self.temp_gradient = widgets.FloatLineEdit('',
                                             label='Temperature Gradient:',
                                             after_label='W/m^2',
                                             layout=hlayout,
                                             readonly=True)
        self.heat_flow_le = widgets.FloatLineEdit('',
                                             label='Heat Flux:',
                                             after_label=u'K/\u212B',
                                             layout=hlayout,
                                             readonly=True)
        self.cross_area_le = widgets.FloatLineEdit('',
                                             label='Cross Sectional Area:',
                                             after_label=u"\u212B<sup>2<sup>",
                                             layout=hlayout)

        hlayout.addStretch(1000)
        self.setMinimumHeight(600)
        # self.statusBar().showMessage('Ready')

    def loadAndDraw(self, file_path=None):
        self.setLogFilePath(file_path=file_path)
        self.setLoadDataLabels()
        self.loadLogFile()
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

    def loadLogFile(self):
        prefix = 'The cross sectional area is'
        with open(self.file_path,'r') as fh:
            line = fh.readline()
            while line:
                if not line.startswith(prefix):
                    line = fh.readline()
                    continue
                line = line[len(prefix):]
                self.cross_area_le.setText(line.split()[0])
                break

    def setLoadDataLabels(self):
        if not self.file_path:
            return

        self.load_data_bn.after_label.setText(os.path.basename(self.file_path))
        self.load_data_bn.after_label.setToolTip(self.file_path)

    def loadData(self):
        if not self.file_path:
            return

        temp_file = self.file_path.replace('-driver.log', '_temp.npz')
        try:
            self.temp_data = np.load(temp_file)['data']
        except FileNotFoundError:
            self.reset()

        ene_file = self.file_path.replace('-driver.log', '_ene.npz')
        try:
            self.ene_data = np.load(ene_file)['data']
        except FileNotFoundError:
            self.reset()

    def draw(self):
        if self.file_path is None:
            return

        self.canvas.plot(self.temp_data, self.ene_data)
        coord_min = self.canvas.temp_lline.position
        coord_max = self.canvas.temp_rline.position
        temp_data = self.canvas.temp_axis.lines[-1].get_xydata()
        sel_temp_data = temp_data[(temp_data[:, 0]>coord_min) & (temp_data[:, 0]<coord_max)]
        temp_gradient, temp_intercept = np.polyfit(sel_temp_data[:, 0], sel_temp_data[:, 1], 1)
        print(temp_gradient)
        self.temp_gradient.setValue(temp_gradient)
        ene_temp_data = self.canvas.ene_axis.lines[-1].get_xydata()
        heat_flow, ene_intercept = np.polyfit(ene_temp_data[:, 0], ene_temp_data[:, 1], 1)
        print(heat_flow)
        self.heat_flow_le.setValue(heat_flow)
        thermal_conductivity = nemd.ThermalConductivity(abs(temp_gradient), heat_flow, self.cross_area_le.value())
        thermal_conductivity.run()
        print(thermal_conductivity.thermal_conductivity)
        self.thermal_conductivity_le.setValue(thermal_conductivity.thermal_conductivity)

    def panel(self):
        self.show()
        sys.exit(self.app.exec_())


def get_panel():
    app = QtWidgets.QApplication(sys.argv)
    panel = NemdPanel(app)
    return panel


def main():
    panel = get_panel()
    panel.panel()


if __name__ == '__main__':
    main()
