import numpy as np
import scipy.optimize as opt

from mappings import rnd_S, center, S_to_fg, S_to_R, DEFAULT_SEED
from fw import make_frank_wolfe_solver
from auxiliary import arrange_distances


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


def ub(X, Y, phi_first=.1, c_first=None, n_restarts=1, center_start=False, max_iter=10,
       tol=1e-8, return_fg=False, verbose=False, rnd=None):
    """
    Find upper bound of dGH(X, Y) by minimizing smoothed dis(R) = dis(f, g) over
    the bi-mapping polytope 𝓢 using Frank-Wolfe.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :param phi_first: upper bound of the non-convexity degree ∈ (0, ½) in the
        first minimization problem (float)
    :param c_first: exponentiation base ∈ (1, ∞) for smoothing the distortion
        in the first minimization problem (float)
    :param center_start: whether to try the center of 𝓢 as a starting point first (bool)
    :param n_restarts: number of restarts for the minimization sequence (int)
    :param max_iter: maximum number of Frank-Wolfe iterations (int)
    :param verbose: whether to print out restarts (bool)
    :return: dGH(X, Y), f [optional], g [optional]
    """
    n, m = len(X), len(Y)

    # Find c for the first minimization if needed.
    if c_first is not None:
        assert c_first > 1, f'starting exponentiation base must be > 1 (c_first={c_first} )'
    else:
        assert phi_first is not None, f'either phi_first or c_start must be given'
        assert 0 < phi_first < .5, f'starting non-convexity UB must be < 0.5 ' \
                                   f'(phi_first={phi_first})'
        c_first = find_c(phi_first, X, Y)

    # Initialize the starting points.
    rnd = rnd or np.random.RandomState(DEFAULT_SEED)
    S0s = [rnd_S(n, m, rnd) for _ in range(n_restarts)]
    if center_start:
        S0s[0] = center(n, m)

    # Find a solution for each of the starting points.
    min_dis_R = np.inf
    fw_seq = []
    for restart, S in enumerate(S0s):
        fw_idx = 0
        c = c_first
        remaining_iter = max_iter
        # Run a sequence of FW solvers using solutions as subsequent warm starts.
        while remaining_iter > 0:
            # Set up next FW solver in the sequence if needed.
            try:
                fw = fw_seq[fw_idx]
            except IndexError:
                fw = make_frank_wolfe_solver(X, Y, c, max_iter=max_iter, tol=tol, verbose=verbose)
                fw_seq.append(fw)

            # Solve the minimization problem.
            S, used_iter = fw(S0=S, max_iter=remaining_iter)

            # Terminate if no iterations were made.
            if used_iter == 0:
                break

            # Move to the next minimization obtained by squaring c.
            remaining_iter -= used_iter
            fw_idx += 1
            c **= 2

        # Project the solution to the set of correspondences and find the
        # resulting distortion.
        R = S_to_R(S, n, m)
        dis_R = dis(R, X, Y)

        # Update the best distortion achieved across all restarts.
        if dis_R < min_dis_R:
            best_f, best_g = S_to_fg(S, n, m)
            min_dis_R = dis_R

        if verbose:
            print(f'finished restart {restart}: ½dis(R)={dis_R/2:.4f}, '
                  f'min ½dis(R)={min_dis_R/2:.4f}')

    res = (min_dis_R/2, best_f, best_g) if return_fg else min_dis_R/2

    return res
