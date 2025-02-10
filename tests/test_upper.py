import pytest
import numpy as np
from scipy import spatial as sp
import dgh


class TestModifiedGromovHausdorff:
    rnd = np.random.RandomState(dgh.DEFAULT_SEED)

    # Distance matrix of a singleton.
    Singleton = np.array([[0]])

    # Distance matrix of a Pythagorean triangle.
    Triangle = np.array([[0, 3, 4],
                         [0, 0, 5],
                         [0, 0, 0]])
    Triangle += Triangle.T

    # Distance matrix of 3d points cloud from standard normal distribution.
    coords_3d = rnd.randn(100, 2)
    Cloud = sp.distance.cdist(coords_3d, coords_3d)

    def test_triangle_vs_singleton(self):
        ub, f, g = dgh.upper(self.Triangle, self.Singleton, rnd=self.rnd, return_fg=True)

        assert ub == np.max(self.Triangle) / 2
        assert f == [0]*len(self.Triangle)

    def test_cloud_vs_singleton(self):
        ub, f, g = dgh.upper(self.Cloud, self.Singleton, rnd=self.rnd, return_fg=True)

        assert ub == np.max(self.Cloud) / 2
        assert f == [0]*len(self.Cloud)

    def test_triangle_isometry(self):
        ub, f, g = dgh.upper(self.Triangle, self.Triangle, rnd=self.rnd, return_fg=True)

        assert ub == 0
        assert f == g == list(range(len(self.Triangle)))

    def test_cloud_isometry(self):
        ub, f, g = dgh.upper(self.Cloud, self.Cloud, rnd=self.rnd, return_fg=True)

        assert ub == 0
        assert f == g == list(range(len(self.Cloud)))
