import os
import pytest
import numpy as np

from nemd import pyfunc
from nemd import fileutils
from nemd import testutils

LAMMPS = 'lammps'
BASE_DIR = testutils.test_file(LAMMPS)


class TestPress:

    @pytest.fixture
    def press(self):
        return pyfunc.Press(os.path.join(BASE_DIR, 'press.data'))

    def testSetData(self, press):
        press.setData()
        assert (999, 2) == press.data.shape

    def testSetAve(self, press):
        press.setData()
        press.setAve()
        np.testing.assert_almost_equal(-46.466, press.ave_press, 3)


class TestBoxLength:

    @pytest.fixture
    def box_length(self):
        return pyfunc.BoxLength(os.path.join(BASE_DIR, 'xyzl.data'))

    def testSetData(self, box_length):
        box_length.setData()
        assert (999, 3) == box_length.data.shape

    def testGetLength(self, box_length):
        box_length.setData()
        np.testing.assert_almost_equal(35.17, box_length.getLength(), 2)


class TestModulus:

    @pytest.fixture
    def modulus(self):
        return pyfunc.Modulus(os.path.join(BASE_DIR, 'press.data'), 100)

    def testAve(self, modulus):
        modulus.setData()
        modulus.setAve()
        assert (100, 6) == modulus.ave.shape

    def testPlot(self, modulus, tmpdir):
        with fileutils.chdir(tmpdir, rmtree=True):
            modulus.setData()
            modulus.setAve()
            modulus.plot()
            assert os.path.isfile('press_modulus.png')

    def testGetModulus(self, modulus, tmpdir):
        with fileutils.chdir(tmpdir, rmtree=True):
            modulus.setData()
            modulus.setAve()
            modulus.setModulus()
            np.testing.assert_almost_equal(1848.86, modulus.modulus, 2)
