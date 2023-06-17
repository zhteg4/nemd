"""

"""
import sys

from nemd import molview
from nemd import widgets
from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets


class TrajPanel(widgets.MainWindow):
    POSITION = 'position'
    DISTANCE = 'distance'
    ANGLE = 'angle'
    DIHEDRAL = 'dihedral'
    MEASURE_COUNT = {POSITION: 1, DISTANCE: 2, ANGLE: 3, DIHEDRAL: 4}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def layOut(self):
        mylayout = self.centralWidget().layout()
        self.browser = QtWebEngineWidgets.QWebEngineView(self)
        frm_vw = molview.FrameView()
        frm_vw.clearData()
        frm_vw.updateLayout()
        lfrm = widgets.Frame()
        self.data_bn = widgets.FileButton('Data File',
                                          after_label='not set',
                                          layout=lfrm.layout(),
                                          orien=widgets.Frame.HORIZONTAL,
                                          filters=['*.data'])
        self.traj_bn = widgets.PushButton('Trajectory',
                                          after_label='not set',
                                          layout=lfrm.layout(),
                                          orien=widgets.Frame.HORIZONTAL)
        self.meas_db = widgets.LabeledComboBox(items=self.MEASURE_COUNT.keys(),
                                               orien=widgets.Frame.HORIZONTAL,
                                               layout=lfrm.layout())
        self.all_frm_bn = widgets.PushButton('Analyze All Frames',
                                             layout=lfrm.layout(),
                                             orien=widgets.Frame.HORIZONTAL)
        lfrm.layout().setAlignment(QtCore.Qt.AlignLeft)
        lfrm.layout().addStretch(1000)
        self.browser.setHtml(frm_vw.fig.to_html(include_plotlyjs='cdn'))
        self.browser.setMinimumHeight(750)
        splitter1 = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter1.addWidget(lfrm)
        splitter1.addWidget(self.browser)
        splitter1.setSizes([200, 800])
        mylayout.addWidget(splitter1)
        mylayout.addStretch(1000)
        self.resize(1000, 800)


    def panel(self):
        self.show()
        sys.exit(self.app.exec())


def get_panel(app=None, argv=None):
    if app is None:
        app = QtWidgets.QApplication(argv)
    panel = TrajPanel(app=app, title='Molecular Trajectory Viewer')
    return panel


def main(argv):
    panel = get_panel(argv=argv)
    panel.panel()


if __name__ == '__main__':
    main(sys.argv[1:])
