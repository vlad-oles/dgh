import numpy as np
import scipy.optimize as opt

from .mappings import rnd_S, center, S_to_fg, S_to_R
from .fw import make_frank_wolfe_solver
from .spaces import diam, rad, arrange_distances
from .constants import DEFAULT_SEED, MAX_C


def find_c(phi, X, Y):
    """
    Find c > 1 s.t. the Hessian is at most φ away from the set of positive
    semidefinite matrices (w.r.t. normalized nuclear norm-induced distance).

    :param phi: upper bound for the degree of non-convexity ∈ (0, ½) (float)
    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :return: c
    """
    n, m = len(X), len(Y)
    N = n + m
    d_max = max(X.max(), Y.max())
    # Quantity for an upper bound on the sum of exponents of the distances.
    p_max = N ** 2 * ((2 ** 1.5 + 4) * np.sqrt(
        N ** 2 * ((X ** 2).sum() + (Y ** 2).sum()) - (X.sum() + Y.sum()) ** 2)) / d_max + 6

    # Function whose roots > 1 are guaranteed to yield the Hessian at most φ away
    # from the set of positive semidefinite matrices.
    def func_to_solve(c):
        # 1-norm of the Hessian.
        sum_H = 8 * ((c**X).sum() + (c**Y).sum() + 2*n*m) * ((c**-X).sum() + (c**-Y).sum() + 2*n*m)

        return 1 - 2 * phi - N / (N + np.sqrt(
            16*N**4 + p_max*(c**(2*d_max) + c**(-2*d_max) - 2) - sum_H**2 / (4 * N**4)))

    # Find a root of the function using secant method with the initial guess at 1.
    c = opt.newton(func_to_solve, 1)
    assert c > 1, f'cannot find c for φ={phi}, consider higher value or providing c_start'

    return c


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


def ub(X, Y, first_phi=.1, first_c=None, iter_budget=100, center_start=False,
       lb=0, tol=1e-8, validate_tri_ineq=False, return_fg=False, verbose=0, rnd=None):
    """
    Find upper bound of dGH(X, Y) by minimizing smoothed dis(R) = dis(f, g) over
    the bi-mapping polytope 𝓢 using Frank-Wolfe.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :param first_phi: upper bound of the non-convexity degree ∈ (0, ½) in the
        first minimization problem (float)
    :param first_c: exponentiation base ∈ (1, ∞) for smoothing the distortion
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

    # Scale all distances to prevent overflow.
    d_max = max(diam_X, diam_Y)
    X, Y = map(lambda Z: Z.copy()/d_max, [X, Y])
    lb /= d_max

    # Find c for the first minimization if needed.
    if first_c is not None:
        assert first_c > 1, f'starting exponentiation base must be > 1 (c_first={first_c} )'
    else:
        assert first_phi is not None, f'either first_phi or first_c must be given'
        assert 0 < first_phi < .5, f'starting non-convexity UB must be < 0.5 ' \
                                   f'(phi_first={first_phi})'
        first_c = find_c(first_phi, X, Y)

    if verbose > 0:
        print(f'using {iter_budget} iterations starting from c={first_c} '
              f'(for X, Y rescaled by {d_max}), dGH≥{lb*d_max}')

    # Find minima from new restarts until run out of iteration budget.
    min_dis_R = np.inf
    fw_seq = []
    restart_idx = 0
    while iter_budget > 0:
        # Initialize new restart.
        S = center(n, m) if restart_idx == 0 and center_start else rnd_S(n, m, rnd)
        restart_iter = 0
        fw_idx = 0
        c = first_c
        solving_for_next_c = True

        # Run a sequence of FW solvers using solutions as subsequent warm starts.
        while solving_for_next_c:
            # Set up next FW solver in the sequence if needed.
            try:
                fw = fw_seq[fw_idx]
            except IndexError:
                fw = make_frank_wolfe_solver(
                    X, Y, c, tol=tol, verbose=verbose)
                fw_seq.append(fw)

            # Solve the minimization problem.
            S, used_iter = fw(S0=S, max_iter=iter_budget)

            # Terminate if no iterations were made, run out of iteration budget,
            # or next c will exceed floating point arithmetic limits.
            solving_for_next_c = 0 < used_iter < iter_budget and c < MAX_C

            # Move to the next minimization obtained by squaring c.
            fw_idx += 1
            with np.errstate(over='ignore'):
                c **= 2
            iter_budget -= used_iter
            restart_iter += used_iter

        # Project the solution to the set of correspondences and find the
        # resulting distortion on the original scale.
        R = S_to_R(S, n, m)
        dis_R = dis(R, X, Y) * d_max

        # Update the best distortion achieved from all restarts.
        if dis_R < min_dis_R:
            best_f, best_g = map(list, S_to_fg(S, n, m))
            min_dis_R = dis_R

        if verbose > 1:
            fg_descr = f' (f={best_f}, g={best_g})' if return_fg else ''
            print(f'restart {restart_idx} (used {restart_iter} iterations): '
                  f'½dis(R)={dis_R/2:.4f}, min ½dis(R)={min_dis_R/2:.4f}{fg_descr}')

        restart_idx += 1

        # Terminate if achieved lower bound.
        if min_dis_R <= lb:
            break

    if verbose > 0:
        print(f'proved dGH≤{min_dis_R/2} after {restart_idx} restarts')

    res = (min_dis_R/2, best_f, best_g) if return_fg else min_dis_R/2

    return res
