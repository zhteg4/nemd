import os
import pytest
from nemd import testutils
import traj_viewer as viewer

DATA_FILE = testutils.test_file(os.path.join('trajs', 'c6.data'))
DUMP_FILE = testutils.test_file(os.path.join('trajs', 'c6.custom'))


class TestApp:
    @pytest.fixture
    def app(self):
        return viewer.App()

    def testSetData(self, app):
        pass
