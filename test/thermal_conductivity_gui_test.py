import os
import sys
import pytest
from unittest import mock
from PyQt5 import QtWidgets

from nemd import testutils
from nemd.testutils import SINGLE_NEMD, CRYSTAL_NEMD

import thermal_conductivity_gui as gui

DRIVER_LOG = testutils.test_file(
    os.path.join(CRYSTAL_NEMD, 'results', 'thermal_conductivity-driver.log'))

APP = QtWidgets.QApplication(sys.argv)


class TestNemdPanel(object):

    @pytest.fixture
    def panel(self):
        return gui.get_panel(app=APP)

    def testSetLogFilePath(self, panel):
        assert panel.log_file is None
        with mock.patch.object(gui, 'os') as os_mock:
            os.path.isfile.return_value = True
            with mock.patch.object(gui, 'QtWidgets') as dlg_mock:
                panel.setLogFilePath(None)
                assert dlg_mock.QFileDialog.called is True
            assert panel.log_file is not None
            panel.setLogFilePath('afilename')
            assert panel.log_file == 'afilename'

    def testSetLoadDataLabels(self, panel):
        assert 'not set' == panel.load_data_bn.after_label.text()
        panel.setLogFilePath(DRIVER_LOG)
        panel.setLoadDataLabels()
        assert 'thermal_conductivity-driver.log' == panel.load_data_bn.after_label.text(
        )

    def testLoadLogFile(self, panel):
        assert panel.cross_area_le.value() is None
        panel.setLogFilePath(DRIVER_LOG)
        panel.loadLogFile()
        panel.setArea()
        assert 855.5744 == panel.cross_area_le.value()

    def testloadData(self, panel):
        panel.setLogFilePath(DRIVER_LOG)
        panel.loadLogFile()
        panel.setArea()
        panel.loadData()
        assert (45, 3) == panel.temp_data.shape
        assert (40000, 3) == panel.ene_data.shape
