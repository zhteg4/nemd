import io
import os
import time
import pytest
import logging
import contextlib
from nemd import testutils
import traj_viewer as viewer
from selenium.webdriver.common.by import By

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
        datafile_lb = dash_duo.wait_for_element("#datafile_lb")
        datafile_lb.text == 'c6.data'
        time.sleep(1)
        assert 23 == len(app.frm_vw.fig.data)
        ele = self.loadFile(dash_duo, tag='traj_input', afile=XYZ_FILE)
        time.sleep(1)
        assert ele.text == ''
        traj_lb = self.getElement(dash_duo, tag='traj_lb')
        assert traj_lb.text == 'c6.xyz'
        assert 23 == len(app.frm_vw.fig.data)

    def testTrajChanged(self, app, dash_duo):
        with contextlib.redirect_stdout(io.StringIO()):
            dash_duo.start_server(app)
        ele = self.loadFile(dash_duo, tag='traj_input', afile=DUMP_FILE)
        assert ele.text == ''
        traj_lb = dash_duo.wait_for_element("#traj_lb")
        assert traj_lb.text == 'c6.custom'
        # assert 1 == len(app.frm_vw.fig.data)
        ele = self.loadFile(dash_duo, tag='datafile_input', afile=DATA_FILE)
        assert ele.text == ''
        datafile_lb = dash_duo.wait_for_element("#datafile_lb")
        assert datafile_lb.text == 'c6.data'
        # Without sleep, fig.data is not updated and the mendeleev complains
        # PytestUnhandledThreadExceptionWarning and SystemExit errors related to
        # cursor.execute(statement, parameters)
        time.sleep(1)
        assert 23 == len(app.frm_vw.fig.data)
        assert 6 == len(app.frm_vw.fig.frames)

    def testMeasureData(self, app, dash_duo):
        with contextlib.redirect_stdout(io.StringIO()):
            dash_duo.start_server(app)
        self.loadFile(dash_duo, tag='datafile_input', afile=DATA_FILE)
        options = self.getOptions(dash_duo)
        options[2].click()
        info_lb = dash_duo.wait_for_element("#measure_info_lb")
        assert info_lb.text.endswith('Select 3 atoms to measure angle')
        options = self.getOptions(dash_duo)
        options[0].click()
        info_lb = dash_duo.wait_for_element("#measure_info_lb")
        assert info_lb.text.endswith('Select 1 atoms to measure position')
        # dash_duo.click_at_coord_fractions(ele, 0.5, 0.6) triggers hovering
        # event instead of data click event

    def getOptions(self, dash_duo):
        ele = self.getElement(dash_duo, tag='measure_dd')
        ele.click()
        menu = ele.find_element(by=By.CSS_SELECTOR,
                                value="div.Select-menu-outer")
        options = menu.find_elements(by=By.CSS_SELECTOR,
                                     value="div.VirtualizedSelectOption")
        return options
