import numpy as np

from .mappings import rnd_S, center, S_to_fg, S_to_R
from .fw import make_frank_wolfe_solver
from .spaces import diam, rad, arrange_distances
from .constants import DEFAULT_SEED, MAX_C


def dis(S, X, Y):
    """
    Calculates "distortion" of a soft mapping pair, which coincides with actual
    distortion on the space of mapping pairs/correspondences.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :return: distortion (float)
    """
    X__Y, Y__X, _Y_X = arrange_distances(X, Y)
    S_Y_X = S @ _Y_X

    dis_S = np.abs(X__Y - S @ Y__X @ S.T + S_Y_X - S_Y_X.T).max()

    return dis_S


def ub(X, Y, c, iter_budget=100, max_iter=-1, center_start=False,
       lb=0, tol=1e-8, validate_tri_ineq=False, return_fg=False, verbose=0, rnd=None):
    """
    Find upper bound of dGH(X, Y) by minimizing smoothed dis(R) = dis(f, g) over
    the bi-mapping polytope 𝓢 using Frank-Wolfe.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :param c: exponentiation base ∈ (1, ∞) for smoothing the distortion
        in the first minimization problem (float)
    :param iter_budget: total number of Frank-Wolfe iterations (int)
    :param center_start: whether to try the center of 𝓢 as a starting point first (bool)
    :param lb: lower bound of dGH(X, Y) to avoid redundant iterations (float)
    :param tol: tolerance to use when evaluating convergence (float)
    :param validate_tri_ineq: whether to validate the triangle inequality (bool)
    :param return_fg: whether to return the optimal pair of mappings (bool)
    :param verbose: no output if 0, summary if >0, restarts if >1, iterations if >2
    :return: dGH(X, Y), f [optional], g [optional]
    """
    # Check that the distances satisfy the metric properties minus the triangle inequality.
    assert (X >= 0).all() and (Y >= 0).all(), 'distance matrices have negative entries'
    assert (np.diag(X) == 0).all() and (np.diag(Y) == 0).all(),\
        'distance matrices have non-zeros on the main diagonal'
    assert (X == X.T).all() and (Y == Y.T).all(), 'distance matrices are not symmetric'
    if validate_tri_ineq:
        assert validate_tri_ineq(X) and validate_tri_ineq(Y),\
            "triangle inequality doesn't hold"

    # Initialize tools for generating starting points.
    n, m = len(X), len(Y)
    rnd = rnd or np.random.RandomState(DEFAULT_SEED)

    # Update lower bound using the radius and diameter differences.
    diam_X, diam_Y = map(diam, [X, Y])
    rad_X, rad_Y = map(rad, [X, Y])
    lb = max(lb, abs(diam_X - diam_Y)/2, abs(rad_X - rad_Y)/2)

    # Scale all distances to avoid overflow.
    d_max = max(diam_X, diam_Y)
    X, Y = map(lambda Z: Z.copy() / d_max, [X, Y])
    lb /= d_max

    if verbose > 0:
        print(f'iteration budget {iter_budget} | c={c} | '
              f'max_iter={max_iter_seq} | dGH≥{lb*d_max}')

    # Find minima from new restarts until iteration budget is depleted.
    min_dis_R = np.inf
    restart_idx = 0
    fw = make_frank_wolfe_solver(X, Y, c, tol=tol, verbose=verbose)
    while iter_budget > 0:
        # Initialize new restart.
        S0 = center(n, m) if restart_idx == 0 and center_start else rnd_S(n, m, rnd)

        # Find new (approximate) solution.
        S, used_iter = fw(S0=S0, max_iter=iter_budget)

        # Update iteration budget.
        iter_budget -= used_iter

        # Project the solution to the set of correspondences and find the
        # resulting distortion on the original scale.
        R = S_to_R(S, n, m)
        dis_R = dis(R, X, Y) * d_max

        # Update the best distortion achieved from all restarts.
        if dis_R < min_dis_R:
            best_f, best_g = map(list, S_to_fg(S, n, m))
            min_dis_R = dis_R

        if verbose > 1:
            fg_descr = f' | f={best_f}, g={best_g}' if return_fg else ''
            print(f'restart {restart_idx} ({used_iter} iterations) | '
                  f'½dis(R)={dis_R/2:.4f} | min ½dis(R)={min_dis_R/2:.4f}{fg_descr}')

        restart_idx += 1

        # Terminate if achieved lower bound.
        if min_dis_R <= lb:
            break

    if verbose > 0:
        print(f'proved dGH≤{min_dis_R/2} after {restart_idx} restarts')

    res = (min_dis_R/2, best_f, best_g) if return_fg else min_dis_R/2

    return res
