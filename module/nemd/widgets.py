# Copyright (c) 2023 The Regents of the Huazhong University of Science and Technology
# All rights reserved.
# This software is licensed under the BSD 3-Clause License.
# Authors: Teng Zhang (2022010236@hust.edu.cn)
"""
This module customizes Qt-related classes.
"""
import os
from PyQt5 import QtCore, QtGui, QtWidgets


class Frame(QtWidgets.QFrame):

    HORIZONTAL = QtCore.Qt.Horizontal
    VERTICAL = QtCore.Qt.Vertical

    def __init__(self, *args, layout=None, orien=VERTICAL, **kwargs):
        super().__init__(*args, **kwargs)
        if layout is not None:
            layout.addWidget(self)
        layout = QtWidgets.QVBoxLayout(
        ) if orien == self.VERTICAL else QtWidgets.QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)


class LabeledComboBox(Frame):

    def __init__(self, items=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        lb = QtWidgets.QLabel('Measure')
        self.layout().addWidget(lb)
        self.comb = QtWidgets.QComboBox()
        if items:
            self.comb.addItems(items)
        self.layout().addWidget(self.comb)
        self.layout().addStretch(1000)


class PushButton(Frame):

    def __init__(self, text, *args, after_label='', alignment=None, command=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.button = QtWidgets.QPushButton(text)
        self.button.sizePolicy().setHorizontalPolicy(
            QtWidgets.QSizePolicy.Policy.Fixed)
        self.layout().addWidget(self.button)
        if command:
            self.button.clicked.connect(command)
        if after_label:
            self.after_label = QtWidgets.QLabel(after_label)
            self.layout().addWidget(self.after_label)
        font = self.button.font()
        font_metrics = QtGui.QFontMetrics(font)
        text_width = font_metrics.averageCharWidth() * len(text)
        text_height = font_metrics.height()
        button_height = self.button.sizeHint().height()
        self.button.setFixedSize(
            QtCore.QSize(text_width + (button_height - text_height) * 2,
                         button_height))
        if alignment:
            self.layout().setAlignment(alignment)


class FileButton(PushButton):

    def __init__(self, *args, filters=None, **kwargs):
        super().__init__(*args, command=self.setFile, **kwargs)
        self.filters = filters
        self.file = None
        self.prev_dir = os.path.curdir
        self.dlg = QtWidgets.QFileDialog(self)
        self.dlg.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        if self.filters:
            self.dlg.setNameFilters(self.filters)

    def setFile(self):
        self.dlg.setDirectory(self.prev_dir)
        if self.dlg.exec():
            self.file = self.dlg.selectedFiles()[0]
            self.prev_dir = os.path.dirname(self.file)
            self.after_label.setText(os.path.basename(self.file))


class LineEdit(Frame):

    def __init__(self,
                 text,
                 label='',
                 after_label='',
                 layout=None,
                 readonly=False,
                 *args,
                 **kwargs):

        self.command = kwargs.pop('command', None)
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        if label:
            self.label = QtWidgets.QLabel(label)
            self.layout.addWidget(self.label)
        self.line_edit = QtWidgets.QLineEdit(text)
        self.line_edit.setFixedWidth(68)
        if readonly:
            self.line_edit.setReadOnly(True)
        self.layout.addWidget(self.line_edit)
        if after_label:
            self.after_label = QtWidgets.QLabel(after_label)
            self.layout.addWidget(self.after_label)
        self.layout.addStretch(1000)
        if self.command:
            self.line_edit.textChanged.connect(self.command)

    def setText(self, text):
        self.line_edit.setText(text)

    def text(self):
        return self.line_edit.text()


class FloatLineEdit(LineEdit):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def value(self):
        try:
            value = float(super().text())
        except ValueError:
            return None
        return value

    def setValue(self, value):
        try:
            value = float(value)
        except TypeError:
            self.line_edit.setText('')
            return

        self.line_edit.setText(f"{value:.6g}")


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, app=None, title=None, **kwargs):
        self.app = app
        self.title = title
        super().__init__(*args, **kwargs)
        self.setWindowTitle()
        self.setCentralWidget()
        self.error_dialog = QtWidgets.QMessageBox(self)
        self.layOut()
        self.addActionButtons()

    def setWindowTitle(self):
        if not self.title:
            return
        super().setWindowTitle(self.title)

    def setCentralWidget(self):
        widget = QtWidgets.QWidget()
        widget.setLayout(QtWidgets.QVBoxLayout())
        super().setCentralWidget(widget)

    def layOut(self):
        pass

    def addActionButtons(self):
        mylayout = self.centralWidget().layout()
        self.reset_bn = PushButton('Reset', layout=mylayout, alignment=QtCore.Qt.AlignRight)
        self.reset_bn.layout().addStretch(1000)

    def error(self, msg):
        self.error_dialog.setText(msg)
        self.error_dialog.exec()