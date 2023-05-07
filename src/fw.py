import numpy as np
from functools import partial

from mappings import fg_to_R
from auxiliary import arrange_distances


def solve_frank_wolfe(obj, grad, find_descent_direction, minimize_obj_wrt_gamma, S0,
                      tol=1e-8, max_iter=10, verbose=0):
    """
    Minimize smoothed distortion over the bi-mapping polytope 𝓢.

    :param obj: smoothed distortion
    :param grad: ∇obj:𝓢🠒𝓢 (function)
    :param find_descent_direction: R:ℝ^(n+m)×(n+m)🠒𝓢 (function)
    :param minimize_obj_wrt_gamma: γ*:𝓢×𝓢🠒ℝ (function)
    :param S0: starting point in 𝓢 (2d-array)
    :param tol: tolerance for measuring rate of descent (float)
    :param max_iter: maximum number of iterations (int)
    :param verbose: :param verbose: {0,1}=no output, 2=print iterations
    :return: solution, number of iterations performed
    """
    S = S0.copy()
    iter = 0
    while iter < max_iter:
        # Find the Frank-Wolfe direction.
        grad_at_S = grad(S)
        R = find_descent_direction(grad_at_S)
        D = R - S

        # Find γ ∈ [0, 1] defining how much to go in the decided direction.
        global_gamma = minimize_obj_wrt_gamma(S, D)
        critical_gammas = {0, 1}
        if 0 < global_gamma < 1:
            critical_gammas.add(global_gamma)
        gamma = min(critical_gammas, key=lambda x: obj(S + x*D))

        if verbose >= 2:
            print(f'  iter {iter}: obj(S)={obj(S):.4f}, γ={gamma:.5f}')

        # Stop if the rate of descent is too small or if the line search stalls.
        if np.sum(-grad_at_S * D) < tol or np.isclose(gamma, 0):
            break

        # Move S by γ towards R, i.e. to (1-γ)S + γR.
        S += gamma * D
        assert np.allclose(np.sum(S, axis=1), 1), \
            f'(1-γ)S + γR is not row-stochastic: S={repr(S - gamma * D)}, D={repr(D)}, γ={gamma}'

        iter += 1

    return S, iter


def make_frank_wolfe_solver(X, Y, c, **kwargs):
    """
    Create Frank-Wolfe solver for minimizing c-smoothed distortion over
    the bi-mapping polytope 𝓢.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :param c: exponentiation base ∈ (1, ∞) for smoothing the distortion (float)
    :param kwargs:
    :return: solver
    """
    n, m = len(X), len(Y)

    # Define auxiliary function that is a component in the objective and its gradient.
    def aux_sum(S):
        X__Y, Y__X, _Y_X = arrange_distances(X, Y)
        c_Y_X, c__Y_X = c**_Y_X,  c**-_Y_X

        return (c__Y_X @ S @ c_Y_X + c_Y_X @ S @ c__Y_X).T + \
            c**-X__Y @ S @ c**Y__X + c**X__Y @ S @ c**-Y__X

    # Smooth distortion as the objective.
    def obj(S):
        return np.sum(S * aux_sum(S)) - 2 * (n + m)**2  # redundant subtraction

    # ∇obj.
    def grad(S):
        return 2 * aux_sum(S)

    # To minimize〈R, ∇obj(S)〉over 𝓢 given S ∈ 𝓢, R must be a vertex of 𝓢.
    def find_descent_direction(grad_at_S):
        f = np.argmin(grad_at_S[:n, :m], axis=1)
        g = np.argmin(grad_at_S[n:, m:], axis=1)

        return fg_to_R(f, g)

    # To minimize obj(γ) = obj(S + γD), for line search.
    def minimize_obj_wrt_gamma(S, D):
        # Leverage that the objective is quadratic in γ, obj(γ) = aγ² + bγ + c.
        a = np.sum(D * aux_sum(D))
        b = np.sum(D * aux_sum(S)) + np.sum(S * aux_sum(D))
        with np.errstate(divide='ignore', invalid='ignore'):
            global_gamma = -b / (2*a)

        return global_gamma

    fw = partial(solve_frank_wolfe, obj, grad, find_descent_direction,
                 minimize_obj_wrt_gamma, **kwargs)

    return fw
