import numpy as np

from constants import DEFAULT_SEED


def rnd_S(n, m, rnd=None):
    """
    Generates random soft mapping in X🠖Y as a point in the bi-mapping polytope 𝓢.

    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :param rnd: NumPy random state
    :return: soft mapping pair S ∈ 𝓢  (2d-array)
    """
    rnd = rnd or np.random.RandomState(DEFAULT_SEED)
    nxn_zeros, mxm_zeros = (np.zeros((size, size)) for size in [n, m])

    # Generate random n×m and m×n row-stochastic matrices.
    F_soft, G_soft = (rnd.rand(size1, size2)
                        for size1, size2 in [(n, m), (m, n)])
    for soft in (F_soft, G_soft):
        soft /= soft.sum(axis=1)[:, None]

    S = np.block([[F_soft, nxn_zeros], [mxm_zeros, G_soft]])

    return S


def center(n, m):
    """
    Returns soft mapping pair that is the barycenter of the bi-mapping polytope 𝓢.

     :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: barycenter of 𝓢  (2d-array)
    """
    nxn_zeros, mxm_zeros = (np.zeros((size, size), dtype=int) for size in [n, m])

    F_center, G_center = (np.full((size1, size2), 1 / size2)
                            for size1, size2 in [(n, m), (m, n)])

    S = np.block([[F_center, nxn_zeros], [mxm_zeros, G_center]])

    return S


def fg_to_R(f, g):
    """
    Represents a mapping pair as a binary row-stochastic block matrix.

    :param f: mapping in X🠖Y (1d-array)
    :param g: mapping in Y🠖X (1d-array)
    :return: mapping pair representation R ∈ ℛ (2d-array)
    """
    n, m = len(f), len(g)
    nxn_zeros, mxm_zeros = (np.zeros((size, size), dtype=int) for size in [n, m])

    # Construct matrix representations of the mappings.
    F = np.identity(m)[f]
    G = np.identity(n)[g]

    R = np.block([[F, nxn_zeros], [mxm_zeros, G]])

    return R


def S_to_fg(S, n, m):
    """
    Projects a soft mapping pair onto the space of mapping pairs.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: projections f:X🠖Y (1d-array), g:Y🠖X (1d-array)
    """
    f = np.argmax(S[:n, :m], axis=1)
    g = np.argmax(S[n:, m:], axis=1)

    return f, g


def S_to_R(S, n, m):
    """
    Projects a soft mapping pair onto the space of mapping pairs and
    represents the projection as a binary row-stochastic block matrix.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: mapping pair representation R ∈ ℛ (2d-array)
    """
    f, g = S_to_fg(S, n, m)
    R = fg_to_R(f, g)

    return R
