import numpy as np
from itertools import permutations


def diam(X):
    """
    Find the diameter of a metric space X.

    :param X: distance matrix of X (2d-array)
    :return: diameter (float)
    """
    return X.max()


def rad(X):
    """
    Find the radius of a metric space X.

    :param X: distance matrix of X (2d-array)
    :return: radius (float)
    """
    return np.min(X.max(axis=0))


def validate_tri_ineq(X):
    """
    Validates that the distance on X obeys the triangle inequality.

    :param X: distance matrix of X (2d-array)
    :return: whether the triangle inequality holds (bool)
    """
    for i, j, k in permutations(range(len(X)), 3):
        if X[i, j] > X[j, k] + X[k, i]:
            return False

    return True


def arrange_distances(X, Y):
    """
    Arrange distances of X and Y in block matrices used in the computations.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :return: three block matrices
    """
    nxm_zeros = np.zeros((len(X), len(Y)), dtype=int)

    X__Y = np.block([[X, nxm_zeros], [nxm_zeros.T, Y]])
    Y__X = np.block([[Y, nxm_zeros.T], [nxm_zeros, X]])
    _Y_X = np.block([[nxm_zeros.T, Y], [X, nxm_zeros]])

    return X__Y, Y__X, _Y_X
