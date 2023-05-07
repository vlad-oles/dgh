# dGH

Computes the Gromov–Hausdorff distance $d_\text{GH}(X, Y)$ by solving (a parametric family of) quadratic minimizations with affine constraints, whose solutions are guaranteed to deliver $d_\text{GH}(X, Y)$ for sufficiently large value of the parameter $c$. The minimizations are solved using the Frank-Wolfe algorithm in $O(n^3)$ time per its iteration, where $n = |X| + |Y|$ is the total number of points. Even when the algorithm fails to find a global minimum, the resulting solution provides an upper bound for $d_\text{GH}(X, Y)$.

A manuscript describing the underlying theory is currently in preparation.