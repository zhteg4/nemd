from PyQt5 import QtCore, QtGui, QtWidgets
import os
import sys
import fileutils
import pytest
import testutils

from testutils import SINGLE_NEMD, CRYSTAL_NEMD

import thermal_conductivity_gui as gui

DRIVER_LOG = testutils.test_file(os.path.join(CRYSTAL_NEMD, 'results', 'thermal_conductivity-driver.log'))


class TestMainWindow(object):

    @pytest.fixture
    def panel(self):
        return gui.get_panel()

    def testLoadAndDraw(self, panel):
        panel.loadAndDraw(file_path=DRIVER_LOG)
        panel.show()
        import pdb;
        pdb.set_trace()
        pass