import numpy as np
from functools import partial
import jax

from .mappings import is_row_stoch, fg_to_R
from .spaces import arrange_distances

def solve_frank_wolfe(obj, grad, find_descent_direction, minimize_obj_wrt_alpha, S0,
                      tol=1e-16, max_iter=np.inf, verbose=0):
    """
    Minimizes smoothed distortion σ over the bi-mapping polytope 𝓢.

    :param obj: smoothed distortion σ:𝓢🠒ℝ (function)
    :param grad: ∇σ:𝓢🠒𝓢 (function)
    :param find_descent_direction: R:ℝ^(n+m)×(n+m)🠒𝓢 (function)
    :param minimize_obj_wrt_alpha: α*:𝓢×𝓢🠒ℝ (function)
    :param S0: starting point in 𝓢 (2d-array)
    :param tol: tolerance for measuring rate of descent (float)
    :param max_iter: maximum number of iterations (int or ∞)
    :param verbose: no output if ≤2, iterations if >2
    :return: solution, number of iterations performed
    """
    S = S0.copy()
    for iter in range(max_iter):
        # Find the Frank-Wolfe direction.
        grad_at_S = grad(S)
        R = find_descent_direction(grad_at_S)
        D = R - S

        # Find α ∈ [0, 1] defining how much to go in the decided direction.
        global_alpha = minimize_obj_wrt_alpha(S, D)
        critical_alphas = {0, 1}
        if 0 < global_alpha < 1:
            critical_alphas.add(global_alpha)
        alpha = min(critical_alphas, key=lambda x: obj(S + x*D))

        if verbose > 2:
            print(f'  iter {iter}: σ(S)={obj(S):.4f}, α={alpha:.5f}')

        # Move S towards R by α, i.e. to (1-α)S + αR.
        S += alpha * D

        # Stop if the rate of descent is too small or if the line search stalls.
        if np.sum(-grad_at_S * D) < tol or np.isclose(alpha, 0):
            break

    return S, iter + 1


def make_frank_wolfe_solver(X, Y, c, **kwargs):
    """
    Creates Frank-Wolfe solver for minimizing c-smoothed distortion over
    the bi-mapping polytope 𝓢.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :param c: exponentiation base ∈ (1, ∞) for smoothing the distortion (float)
    :return: solver
    """
            
    n, m = len(X), len(Y)
    X__Y, Y__X, _Y_X = arrange_distances(X, Y)

    # Precompute the exponentials.
    exp__Y_X, exp_neg__Y_X = c**_Y_X, c**-_Y_X
    exp_X__Y, exp_neg_X__Y = c**X__Y, c**-X__Y
    exp_Y__X, exp_neg_Y__X = c**Y__X, c**-Y__X

    @jax.jit
    def jax_matrix_multiply(S, exp__Y_X, exp_neg__Y_X, exp_X__Y, exp_neg_X__Y,
                            exp_Y__X, exp_neg_Y__X):
        return (exp_neg__Y_X @ S @ exp__Y_X + exp__Y_X @ S @ exp_neg__Y_X).T + \
           exp_neg_X__Y @ S @ exp_Y__X + exp_X__Y @ S @ exp_neg_Y__X
    
    # Define auxiliary function that is a component in the objective and its gradient.
    def dot_multiplicand(S):
        # Use precomputed values and precompiled jax function.
        return jax_matrix_multiply(S, exp__Y_X, exp_neg__Y_X, exp_X__Y, exp_neg_X__Y,
                            exp_Y__X, exp_neg_Y__X)

    # Smooth distortion σ as the objective.
    def obj(S):
        return np.sum(S * dot_multiplicand(S))

    # ∇σ.
    def grad(S):
        return 2 * dot_multiplicand(S)

    # To minimize〈R, ∇σ(S)〉over 𝓢 given S ∈ 𝓢, R must be a vertex of 𝓢.
    def find_descent_direction(grad_at_S):
        f = np.argmin(grad_at_S[:n, :m], axis=1)
        g = np.argmin(grad_at_S[n:, m:], axis=1)

        return fg_to_R(f, g)

    # To minimize σ(α) = σ(S + αD), for line search.
    def minimize_obj_wrt_alpha(S, D):
        # Leverage that the objective is quadratic in α, σ(α) = aα² + bα + c.
        a = np.sum(D * dot_multiplicand(D))
        b = np.sum(D * dot_multiplicand(S)) + np.sum(S * dot_multiplicand(D))
        with np.errstate(divide='ignore', invalid='ignore'):
            global_alpha = np.divide(-b, 2*a)

        return global_alpha

    fw = partial(solve_frank_wolfe, obj, grad, find_descent_direction,
                 minimize_obj_wrt_alpha, **kwargs)

    return fw
