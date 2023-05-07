import numpy as np


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
