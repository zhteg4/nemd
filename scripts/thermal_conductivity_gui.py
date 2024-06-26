# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This graphical user interface virtualizes temperature profile and energy flow,
allowing users to recalculate thermal conductivity by customizing temperature
range and heat flux using non-equilibrium molecular dynamics method.
"""
import os
import re
import sys
import numpy as np
from types import SimpleNamespace
from PyQt5 import QtWidgets
from matplotlib import lines
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from nemd import nemd_tc
from nemd import widgets
from nemd import fileutils

COLORS = dict(colors.BASE_COLORS, **colors.CSS4_COLORS)


class DraggableLine(object):

    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'

    def __init__(self, axis, position=0.1, command=None, kind=VERTICAL):
        self.axis = axis
        self.kind = kind
        self.position = position
        self.command = command

        self.canvas = self.axis.get_figure().canvas

        x_line = [position, position]
        y_line = [-1, 1]
        if kind == self.HORIZONTAL:
            x_line, y_line = y_line, x_line
        self.line = lines.Line2D(x_line,
                                 y_line,
                                 color=COLORS['orange'],
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
        if self.command:
            self.command()

    def setPosition(self):
        if self.kind == self.HORIZONTAL:
            self.position = self.line.get_ydata()[0]
        else:
            self.position = self.line.get_xdata()[0]


class LineWithVSpan(DraggableLine):

    LEFT = 'left'
    RIGHT = 'right'

    def __init__(self, *args, fill_direction='right', line_lim=None, **kwargs):
        self.fill_direction = fill_direction
        self.line_lim = line_lim
        super().__init__(*args, **kwargs)
        xrange = list(self.axis.get_xlim())
        edge_index = 0 if fill_direction == self.RIGHT else 1
        xrange[edge_index] = self.position
        self.polygon = self.axis.axvspan(*xrange, alpha=0.3, color='grey')

    def followmouse(self, event):
        if event.xdata == None:
            return
        self.modifyEvent(event)
        self.setLinePosition(event)
        self.resizeVSpan(event)
        self.canvas.draw_idle()

    def modifyEvent(self, event):

        if len(self.axis.lines) <= 2:
            return

        if event.xdata == None:
            return

        xdata = self.axis.lines[-1].get_xdata()
        x_min, x_max = min(xdata), max(xdata)

        if event.xdata < x_min:
            event.xdata = x_min
        elif event.xdata > x_max:
            event.xdata = x_max

        if not self.line_lim:
            return event
        if self.fill_direction == 'right':
            avail_xdata = xdata[xdata > self.line_lim.position]
            if len(avail_xdata[avail_xdata < event.xdata]) < 3:
                avail_min, avail_min2, avail_min3 = np.partition(
                    avail_xdata, 3)[:3]
                event.xdata = (avail_min2 + avail_min3) / 2
        else:
            avail_xdata = xdata[xdata < self.line_lim.position]
            if len(avail_xdata[avail_xdata > event.xdata]) < 3:
                avail_max3, avail_max2, avail_max = np.partition(
                    avail_xdata, -3)[-3:]
                event.xdata = (avail_max2 + avail_max3) / 2
        return event

    def update(self, event):
        self.resizeVSpan(event)
        super().update(event)
        self.setPosition()

    def resizeVSpan(self, event):
        vpan_xy = self.polygon.get_xy()
        xmin, xmax = self.axis.get_xlim()
        x_lim = xmax if self.fill_direction == self.RIGHT else xmin
        vpan_xy[:, 0] = [event.xdata, event.xdata, x_lim, x_lim, event.xdata]
        self.polygon.set_xy(vpan_xy)


class Canvas(FigureCanvasQTAgg):

    def __init__(self, width=5, height=4, dpi=100, command=None):
        self.command = command
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.temp_axis = self.fig.add_subplot(211)
        self.ene_axis = self.fig.add_subplot(212)
        self.setUpAxises()
        self.temp_lline = LineWithVSpan(self.temp_axis,
                                        fill_direction=LineWithVSpan.LEFT,
                                        command=self.dataRangeChanged)
        self.temp_rline = LineWithVSpan(self.temp_axis,
                                        position=0.9,
                                        command=self.dataRangeChanged,
                                        line_lim=self.temp_lline)
        self.temp_lline.line_lim = self.temp_rline
        self.fig.tight_layout()

        self.temp_gradient = None
        self.heat_flow = None
        self.temp_line = None
        self.temp_fit = None
        self.temp_poly = None
        self.ene_line = None
        self.ene_poly = None

    def dataRangeChanged(self):
        if len(self.temp_axis.lines) <= 2:
            return
        self.setGradients()
        self.plotFit()
        if self.command:
            self.command()

    def setGradients(self):

        coord_min = self.temp_lline.position
        coord_max = self.temp_rline.position
        temp_data = self.temp_line.get_xydata()
        self.sel_temp_data = temp_data[(temp_data[:, 0] > coord_min)
                                       & (temp_data[:, 0] < coord_max)]
        self.temp_gradient, self.temp_intercept = np.polyfit(
            self.sel_temp_data[:, 0], self.sel_temp_data[:, 1], 1)

        ene_temp_data = self.ene_axis.lines[-1].get_xydata()
        self.heat_flow, ene_intercept = np.polyfit(ene_temp_data[:, 0],
                                                   ene_temp_data[:, 1], 1)

    def plotFit(self):
        fitted = np.polyval([self.temp_gradient, self.temp_intercept],
                            self.sel_temp_data[:, 0])
        if self.temp_fit is None:
            self.temp_fit = lines.Line2D(self.sel_temp_data[:, 0],
                                         fitted,
                                         dashes=[6, 2],
                                         label='Fitted',
                                         c='r')
            self.temp_axis.add_line(self.temp_fit)
        else:
            self.temp_fit.set_data(self.sel_temp_data[:, 0], fitted)

        self.temp_axis.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        self.ene_axis.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        self.fig.tight_layout()
        self.draw_idle()

    def setUpAxises(self):
        self.temp_axis.set_xlim((0, 1))
        self.temp_axis.set_xlabel(u"Coordinates (\u212B)")
        self.temp_axis.set_ylabel('Temperature (K)')
        self.ene_axis.ticklabel_format(style='scientific',
                                       scilimits=(-3, 3),
                                       useMathText=True)
        self.ene_axis.set_xlabel("Time (ns)")
        self.ene_axis.set_ylabel('Energy (Kcal/mol)')

    def plotData(self, temp_data, ene_data):

        self.plotTempLines(temp_data)
        self.plotTemp(temp_data)
        self.plotEne(ene_data)

    def plotTempLines(self,
                      temp_data,
                      lim_frac=(-0.1, 1.1),
                      line_frac=(0.1, 0.9)):

        coordinates = temp_data[:, 0]
        temp = temp_data[:, 1]
        coord_min = min(coordinates)
        coord_span = max(coordinates) - coord_min
        temp_min = min(temp)
        temp_span = max(temp) - temp_min

        xlims = [x * coord_span + coord_min for x in lim_frac]
        self.temp_axis.set_xlim(*xlims)
        ylims = [x * temp_span + temp_min for x in lim_frac]
        self.temp_axis.set_ylim(*ylims)

        line_events = [
            SimpleNamespace(xdata=x * coord_span + coord_min)
            for x in line_frac
        ]
        self.temp_lline.update(line_events[0])
        self.temp_rline.update(line_events[1])

    def plotTemp(self, temp_data):
        coordinates = temp_data[:, 0]
        temp = temp_data[:, 1]
        temp_lower_bound = temp - temp_data[:, 2]
        temp_upper_bound = temp + temp_data[:, 2]

        if self.temp_line is None:
            self.temp_line = lines.Line2D(coordinates,
                                          temp,
                                          label='Mean',
                                          c='k')
            self.temp_axis.add_line(self.temp_line)
        else:
            self.temp_line.set_data(coordinates, temp)
        if self.temp_poly is not None:
            self.temp_axis.collections.remove(self.temp_poly)
        self.temp_poly = self.temp_axis.fill_between(
            coordinates,
            temp_lower_bound,
            temp_upper_bound,
            alpha=0.2,
            label='Standard deviation',
            color='b')

    def plotEne(self, ene_data):
        time = ene_data[:, 0]
        energy = ene_data[:, 1]
        energy_lower_bound = energy - ene_data[:, 2]
        energy_upper_bound = energy + ene_data[:, 2]
        if self.ene_line is None:
            self.ene_line = lines.Line2D(time, energy, label='Mean', c='k')
            self.ene_axis.add_line(self.ene_line)
        else:
            self.ene_line.set_data(time, energy)

        if self.ene_poly is not None:
            self.ene_axis.collections.remove(self.ene_poly)
        self.ene_poly = self.ene_axis.fill_between(time,
                                                   energy_lower_bound,
                                                   energy_upper_bound,
                                                   alpha=0.2,
                                                   label='Standard deviation',
                                                   color='b')


class QMainWindow(QtWidgets.QMainWindow):

    def __init__(self, app, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)
        self.error_dialog = QtWidgets.QMessageBox(self)
        # self.error_dialog.setIcon(QtWidgets.QMessageBox.critical)

    def error(self, msg):
        self.error_dialog.setText(msg)
        self.error_dialog.exec()


class NemdPanel(QMainWindow):

    def __init__(self, app, *args, **kwargs):
        super(NemdPanel, self).__init__(app, *args, **kwargs)
        self.setWindowTitle('Thermal Conductivity Viewer')
        self.central_layout = QtWidgets.QVBoxLayout()
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(self.central_layout)
        self.setCentralWidget(central_widget)

        self.load_data_bn = widgets.PushButton('Load Data',
                                               after_label='not set',
                                               layout=self.central_layout,
                                               command=self.loadAndDraw)

        self.canvas = Canvas(command=self.setGradients)
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
        self.temp_gradient_le = widgets.FloatLineEdit(
            '',
            label='Temperature Gradient:',
            after_label='W/m^2',
            layout=hlayout,
            readonly=True,
            command=self.setThermalConductivity)
        self.heat_flow_le = widgets.FloatLineEdit(
            '',
            label='Heat Flux:',
            after_label=u'K/\u212B',
            layout=hlayout,
            readonly=True,
            command=self.setThermalConductivity)
        self.cross_area_le = widgets.FloatLineEdit(
            '',
            label='Cross Sectional Area:',
            after_label=u"\u212B<sup>2<sup>",
            layout=hlayout,
            command=self.setThermalConductivity)

        hlayout.addStretch(1000)
        self.setMinimumHeight(600)
        # self.statusBar().showMessage('Ready')
        self.log_file = None
        self.dirver_log_lines = None
        self.area_line_index = None
        self.previous_dir = os.path.curdir

    def loadAndDraw(self, log_file=None):
        self.setLogFilePath(log_file=log_file)
        self.setLoadDataLabels()
        self.loadLogFile()
        self.setArea()
        self.loadData()
        self.draw()

    def setLogFilePath(self, log_file=None):
        if not log_file:
            dlg = QtWidgets.QFileDialog(self)
            dlg.setDirectory(self.previous_dir)
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
            dlg.setNameFilters(["Driver log (*-driver.log)"])
            # Printed to terminal: python3[16731:2156112] +
            # [CATransaction synchronize] called within transaction
            # Sounds like a macos Ventura  issue:
            # https://forum.qt.io/topic/141868/macos-catransaction-synchronize-called-within-transaction
            # https://github.com/python/cpython/issues/101895
            if dlg.exec():
                log_file = dlg.selectedFiles()[0]
        if not log_file or not os.path.isfile(log_file):
            self.error('Please select a driver.log file.')
            return

        self.log_file = log_file
        self.previous_dir = os.path.dirname(self.log_file)

    def loadLogFile(self):
        if self.log_file is None:
            return

        with open(self.log_file, 'r') as fh:
            self.dirver_log_lines = fh.readlines()

    def setArea(self):
        if self.log_file is None:
            return

        re_area_compiled = re.compile(fileutils.REX_AREA)
        matched_line_gn = (index
                           for index, line in enumerate(self.dirver_log_lines)
                           if re_area_compiled.findall(line))
        self.area_line_index = next(matched_line_gn)
        cross_area = re_area_compiled.findall(
            self.dirver_log_lines[self.area_line_index])[0]
        self.cross_area_le.setText(cross_area)

    def setLoadDataLabels(self):
        if not self.log_file:
            return

        self.load_data_bn.after_label.setText(os.path.basename(self.log_file))
        self.load_data_bn.after_label.setToolTip(self.log_file)

    def loadData(self):
        if not self.log_file:
            return

        temp_file = self.log_file.replace('-driver.log', '_temp.npz')
        try:
            self.temp_data = np.load(temp_file)['data']
        except FileNotFoundError:
            self.reset()

        ene_file = self.log_file.replace('-driver.log', '_ene.npz')
        try:
            self.ene_data = np.load(ene_file)['data']
        except FileNotFoundError:
            self.reset()

    def draw(self):
        if self.log_file is None:
            return

        self.canvas.plotData(self.temp_data, self.ene_data)
        self.canvas.dataRangeChanged()

    def setThermalConductivity(self):
        thermal_conductivity = nemd_tc.ThermalConductivity(
            self.temp_gradient_le.value(), self.heat_flow_le.value(),
            self.cross_area_le.value())
        thermal_conductivity.run()
        self.thermal_conductivity_le.setValue(
            thermal_conductivity.thermal_conductivity)

    def setGradients(self):
        self.temp_gradient_le.setValue(self.canvas.temp_gradient)
        self.heat_flow_le.setValue(self.canvas.heat_flow)

    def panel(self):
        self.show()
        sys.exit(self.app.exec())


def get_panel(app=None):
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    panel = NemdPanel(app)
    return panel


def main():
    panel = get_panel()
    panel.panel()


if __name__ == '__main__':
    main()
