from PyQt5 import QtCore, QtGui, QtWidgets


class PushButton(QtWidgets.QFrame):
    def __init__(self,
                 text,
                 label='',
                 after_label='',
                 layout=None,
                 readonly=False,
                 command=None,
                 *args,
                 **kwargs):

        super().__init__()
        layout.addWidget(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        self.button = QtWidgets.QPushButton(text)
        self.button.sizePolicy().setHorizontalPolicy(
            QtWidgets.QSizePolicy.Fixed)
        self.layout.addWidget(self.button)
        if command:
            self.button.clicked.connect(command)
        if after_label:
            self.after_label = QtWidgets.QLabel(after_label)
            self.layout.addWidget(self.after_label)

        font = self.button.font()
        font_metrics = QtGui.QFontMetrics(font)
        text_width = font_metrics.width(text)
        text_height = font_metrics.height()
        button_height = self.button.sizeHint().height()
        self.button.setFixedSize(
            QtCore.QSize(text_width + (button_height - text_height) * 2,
                         button_height))


class LineEdit(QtWidgets.QFrame):
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
        layout.addWidget(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        if label:
            self.label = QtWidgets.QLabel(label)
            self.layout.addWidget(self.label)
        self.line_edit = QtWidgets.QLineEdit(text)
        self.line_edit.setFixedWidth(65)
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
