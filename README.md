# dGH

Computes the Gromov–Hausdorff distance $$d_\text{GH}(X, Y) = \frac{1}{2}\inf_{f:X\to Y, g:Y\to X} \text{dis}\Big(\\{(x, f(x)): x \in X\\} \cup \\{(g(y), y): y \in Y\\}\Big),$$ where $\text{dis}(R) = \sup_{(x, y), (x', y') \in R} |d_X(x, x') - d_Y(y, y')|$ for $R \subseteq X \times Y,$ by solving its parametric relaxation whose solutions are guaranteed to deliver $d_\text{GH}(X, Y)$ for sufficiently large value of the parameter $c$. The quadratic relaxation with affine constraints is solved using the Frank–Wolfe algorithm in $O(n^3)$ time per its iteration, where $n = |X| + |Y|$ is the total number of points. Even when the algorithm fails to find a global minimum, the resulting solution provides an upper bound for $d_\text{GH}(X, Y)$.

A manuscript describing the underlying theory is currently in preparation.

## Quickstart

Installing the package from Python Package Index:

```$ pip install dgh```

Computing $d_\text{GH}(X, Y)$ where $X$ is the vertices of a tall narrow rectangle and $Y$ is the vertices of a small equilateral triangle together with a remote point from it (see illustration):

<p align="center">
    <img src="https://github.com/vlad-oles/dgh/blob/main/illustration.svg" alt="Illustration of the example" width="300"/>
</p>

```
import numpy as np
import dgh

# Set distance matrix for the rectangle.
X = np.array([[0, 1, 10, 10],
              [0, 0, 10, 10],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
X += X.T

# Set distance matrix for the triangle and a remote point.
Y = np.array([[0, 1, 1, 10],
              [0, 0, 1, 10],
              [0, 0, 0, 10],
              [0, 0, 0, 0]])
Y += Y.T

# Find an upper bound of the Gromov–Hausdorff distance. In this case, dGH(X, Y) = 0.5.
dGH = dgh.ub(X, Y)
```

Increasing the budget of Frank–Wolfe iterations, and thus the number of restarts, allocated for the search (the default budget is 100):

```dGH = dgh.ub(X, Y, iter_budget=1000)```

Obtaining the mappings $f:X\to Y, g:Y\to X$ (as arrays s.t. $f_i = j \Leftrightarrow f(x_i) = y_j$) that deliver the found minimum:

```dGH, f, g = dgh.ub(X, Y, return_fg=True)```

## Advanced
The performance can be improved by explicitly specifying the relaxation parameter $c \in (1, \infty)$. Small $c$ makes the relaxation easier to solve, but its solutions are more likely to deliver the Gromov–Hausdorff distance when $c$ is large.

By default, the method allocates half of the iteration budget to select the best value of $c$ for $(X, Y)$ from $1+10^{-4}, 1+10^{-2},\ldots,1+10^8$, and then spends the remaining half on refining the Gromov–Hausdorff distance using this $c$. Revealing the value of $c$ selected after the search:

```dgh.ub(X, Y, verbose=1)```

Explicitly providing $c$ to the method (to avoid search and/or to test for better accuracy):

```dGH = dgh.ub(X, Y, c=my_c)```


## Contributing
If you found a bug or want to suggest an enhancement, you can create a [GitHub Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). Alternatively, you can email vlad.oles (at) proton (dot) me.

## License
dGH is released under the MIT license.
