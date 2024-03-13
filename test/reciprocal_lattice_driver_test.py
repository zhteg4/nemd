import os
import pytest
import numpy as np

from nemd import fileutils
import reciprocal_lattice_driver as driver


class TestReciprocal:

    @pytest.fixture
    def reciprocal(self, tmpdir, tmp_path):
        return driver.Reciprocal(None)

    def testSetRealVectors(self, reciprocal):
        reciprocal.setRealVectors()
        np.testing.assert_almost_equal([1.5, 0.8660254], reciprocal.a_vect)
        np.testing.assert_almost_equal([1.5, -0.8660254], reciprocal.b_vect)

    def testSetReciprocalVectors(self, reciprocal):
        reciprocal.setRealVectors()
        reciprocal.setReciprocalVectors()
        np.testing.assert_almost_equal([2.0943951, 3.62759873],
                                       reciprocal.ga_vect)
        np.testing.assert_almost_equal([2.0943951, -3.62759873],
                                       reciprocal.gb_vect)

    @pytest.mark.parametrize(('miller_indices'), [([0, 1]), ([1, 0]),
                                                  ([2, 11])])
    def testRun(self, reciprocal, miller_indices, tmpdir):
        with fileutils.chdir(tmpdir, rmtree=True):
            argv = [driver.FLAG_MILLER_INDICES
                    ] + [str(x) for x in miller_indices]
            reciprocal.options = driver.validate_options(argv)
            reciprocal.run()
            assert os.path.isfile('reciprocal_lattice.png')
