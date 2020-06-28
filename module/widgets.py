from PyQt5 import QtCore, QtGui, QtWidgets


class LineEdit(QtWidgets.QFrame):
    def __init__(self,
                 text,
                 label='',
                 after_label='',
                 layout=None,
                 readonly=False,
                 *args,
                 **kwargs):

        super().__init__()
        layout.addWidget(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)
        if label:
            self.label = QtWidgets.QLabel('Thermal Conductivity:')
            self.layout.addWidget(self.label)
        self.line_edit = QtWidgets.QLineEdit('100')
        self.line_edit.setFixedWidth(60)
        if readonly:
            self.line_edit.setReadOnly(True)
        self.layout.addWidget(self.line_edit)
        if after_label:
            self.after_label = QtWidgets.QLabel('W/(m⋅K)')
            self.layout.addWidget(self.after_label)
        self.layout.addStretch(1000)
