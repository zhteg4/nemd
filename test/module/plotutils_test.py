import math
import pytest
import numpy as np

from nemd import plotutils


class TestLatticePlotter:

    @pytest.fixture
    def plotter(self, indices):
        with plotutils.get_pyplot() as plt:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            charac_length = math.sqrt(
                3)  # sqrt(3) X the edge length of hexagon
            a_vect = np.array([math.sqrt(3) / 2., 0.5]) * charac_length
            b_vect = np.array([math.sqrt(3) / 2., -0.5]) * charac_length
            return plotutils.LatticePlotter(ax,
                                            a_vect=a_vect,
                                            b_vect=b_vect,
                                            indices=indices)

    @pytest.mark.parametrize(('indices', 'num'), [([0, 1], 145), ([1, 0], 145),
                                                  ([2, 11], 361)])
    def testSetGrids(self, plotter, num):
        plotter.setGrids()
        assert num == len(plotter.xs)

    @pytest.mark.parametrize(
        ('indices', 'ma_norm', 'mb_norm'),
        [([0, 1], 0, 1.7320508075688772), ([1, 0], 1.7320508075688772, 0),
         ([2, 11], 3.4641016151377544, 19.052558883257646)])
    def testSetVects(self, plotter, ma_norm, mb_norm):
        plotter.setVects()
        assert ma_norm == np.linalg.norm(plotter.ma_vect)
        assert mb_norm == np.linalg.norm(plotter.mb_vect)

    @pytest.mark.parametrize(
        ('indices', 'index', 'pnt1', 'pnt2'),
        [([0, 1], 1, [12., 5.19615242], [-9., -6.92820323]),
         ([1, 0], 1, [12., -5.19615242], [-9., 6.92820323]),
         ([2, 11], 1, [18.57692308, -11.25833025], [-8.42307692, 11.25833025]),
         ([0, 1], 0, [-12., -6.92820323], [12., 6.92820323]),
         ([1, 0], 0, [-12., 6.9282032], [12., -6.9282032]),
         ([2, 11], 0, [13.5, -11.2583302], [-13.5, 11.2583302])])
    def testSelectPoints(self, plotter, index, pnt1, pnt2):
        plotter.setGrids()
        plotter.setVects()
        pnts = plotter.selectPoints(index=index)
        np.testing.assert_almost_equal(pnt1, pnts[0])
        np.testing.assert_almost_equal(pnt2, pnts[1])

    @pytest.mark.parametrize(('indices', 'index', 'vect'),
                             [([0, 1], 1, [0.75, -1.29903811]),
                              ([1, 0], 1, [0.75, 1.29903811]),
                              ([2, 11], 1, [2.08252427, 2.49718005]),
                              ([0, 1], -1, [-0.75, 1.29903811]),
                              ([1, 0], -1, [-0.75, -1.29903811]),
                              ([2, 11], -1, [-2.08252427, -2.49718005])])
    def testSelectPoints(self, plotter, index, vect):
        plotter.setGrids()
        plotter.setVects()
        nvect = plotter.getDPoint(index=index)
        np.testing.assert_almost_equal(vect, nvect)


class TestReciprocalLatticePlotter:

    @pytest.fixture
    def plotter(self, indices):
        with plotutils.get_pyplot() as plt:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            charac_length = math.sqrt(
                3)  # sqrt(3) X the edge length of hexagon
            a_vect = np.array([math.sqrt(3) / 2., 0.5]) * charac_length
            b_vect = np.array([math.sqrt(3) / 2., -0.5]) * charac_length
            return plotutils.ReciprocalLatticePlotter(ax,
                                                      a_vect=a_vect,
                                                      b_vect=b_vect,
                                                      indices=indices)

    @pytest.mark.parametrize(('indices', 'rindices'),
                             [([0, 1], [0, 1]), ([1, 0], [1, 0]),
                              ([2, 11], [0.5, 0.09090909090909091])])
    def testSetGrids(self, plotter, rindices):
        plotter.setIndexes()
        np.testing.assert_almost_equal(rindices, plotter.indices)
