import io
import os
import time
import pytest
import logging
import contextlib
from nemd import testutils
import traj_viewer as viewer

DATA_FILE = testutils.test_file(os.path.join('trajs', 'c6.data'))
DUMP_FILE = testutils.test_file(os.path.join('trajs', 'c6.custom'))
XYZ_FILE = testutils.test_file(os.path.join('trajs', 'c6.xyz'))


class TestApp:

    XPATH = 'XPATH'

    @pytest.fixture
    def app(self):
        app = viewer.App(__name__)
        app.logger.setLevel(logging.WARNING)
        return app

    def loadFile(self, dash_duo, tag, afile):
        ele = self.getElement(dash_duo, tag=tag, input=True)
        ele.send_keys(os.path.normpath(afile))
        return ele

    def getElement(cls, dash_duo, xpath=None, tag=None, input=False):
        if xpath is None:
            xpath = f'//*[@id="{tag}"]'
        if input:
            xpath += '/div/input'
        return dash_duo.find_element(xpath, attribute=cls.XPATH)

    def testDataFileChanged(self, app, dash_duo):
        with contextlib.redirect_stdout(io.StringIO()):
            dash_duo.start_server(app)
        ele = self.loadFile(dash_duo, tag='datafile_input', afile=DATA_FILE)
        assert ele.text == ''
        datafile_lb = self.getElement(dash_duo, tag='datafile_lb')
        assert datafile_lb.text == 'c6.data'
        time.sleep(1)
        assert 11 == len(app.frm_vw.fig.data)
        ele = self.loadFile(dash_duo, tag='traj_input', afile=XYZ_FILE)
        time.sleep(1)
        assert ele.text == ''
        traj_lb = self.getElement(dash_duo, tag='traj_lb')
        assert traj_lb.text == 'c6.xyz'
        assert 11 == len(app.frm_vw.fig.data)

    def testTrajChanged(self, app, dash_duo):
        with contextlib.redirect_stdout(io.StringIO()):
            dash_duo.start_server(app)
        ele = self.loadFile(dash_duo, tag='traj_input', afile=DUMP_FILE)
        assert ele.text == ''
        datafile_lb = self.getElement(dash_duo, tag='traj_lb')
        assert datafile_lb.text == 'c6.custom'
        assert 1 == len(app.frm_vw.fig.data)
        ele = self.loadFile(dash_duo, tag='datafile_input', afile=DATA_FILE)
        assert ele.text == ''
        datafile_lb = self.getElement(dash_duo, tag='datafile_lb')
        assert datafile_lb.text == 'c6.data'
        # Without sleep, fig.data is not updated and the mendeleev complains
        # PytestUnhandledThreadExceptionWarning and SystemExit errors related to
        # cursor.execute(statement, parameters)
        time.sleep(1)
        assert 11 == len(app.frm_vw.fig.data)
        assert 5 == len(app.frm_vw.fig.frames)
