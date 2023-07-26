# dGH

Given the distance matrices of some metric spaces $X$ and $Y$, estimates the Gromov–Hausdorff distance $$d_\text{GH}(X, Y) = \frac{1}{2}\inf_{f:X\to Y, g:Y\to X} \text{dis}\Big(\\{(x, f(x)): x \in X\\} \cup \\{(g(y), y): y \in Y\\}\Big),$$ where $$\text{dis}(R) = \sup_{(x, y), (x', y') \in R} |d_X(x, x') - d_Y(y, y')|$$ for $R \subseteq X \times Y$.

The distance is estimated from above by solving its parametric relaxation whose solutions are guaranteed to deliver $d_\text{GH}(X, Y)$ for sufficiently large value of the parameter $c$. The quadratic relaxation with affine constraints is solved using the Frank–Wolfe algorithm in $O(n^3)$ time per its iteration, where $n = \max\{|X|, |Y|\}$ is the number of points in the larger space. Even if the algorithm fails to find $d_\text{GH}(X, Y)$ exactly, the resulting solution provides its upper bound.

A detailed description of the relaxation, its optimality guarantees and optimization landscape, and the approach to solving it can be found in [Computing the Gromov–Hausdorff distance using first-order methods](https://arxiv.org/pdf/2307.13660.pdf).

## Quickstart

To install the package from Python Package Index:

```$ pip install dgh```

Consider an example in which $X$ is comprised by the vertices of a $1 \times 10$ rectangle and $Y$ — by the vertices of a unit equilateral triangle together with a point that is 10 away fom them (see illustration).

<p align="center">
    <img src="https://github.com/vlad-oles/dgh/blob/main/illustration.svg" alt="Illustration of the example" width="300"/>
</p>

To create their distance matrices (in which the (i,j)-th entry contains the distance between the i-th and j-th points of the space):

```
import numpy as np

X = np.array([[0, 1, 10, 10],
              [0, 0, 10, 10],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
X += X.T

Y = np.array([[0, 1, 1, 10],
              [0, 0, 1, 10],
              [0, 0, 0, 10],
              [0, 0, 0, 0]])
Y += Y.T
```

To compute (an upper bound of) their Gromov--Hausdorff distance $d_\text{GH}(X, Y)$:

```
import dgh

dGH = dgh.upper(X, Y)
```

In this case, the distance is computed exactly as $d_\text{GH}(X, Y)=\frac{1}{2}$. 

## Basics

By default, the computational budget allocated for the search is 100 Frank–Wolfe iterations. Bigger budget means more random restarts (and/or better convergence) and therefore the accuracy. To set the budget:

```dGH = dgh.upper(X, Y, iter_budget=my_budget)```

To obtain the mappings $f:X\to Y, g:Y\to X$ (as arrays s.t. $f_i = j \Leftrightarrow f(x_i) = y_j$) that deliver the found minimum:

```dGH, f, g = dgh.upper(X, Y, return_fg=True)```

## Advanced
The performance can be improved by explicitly specifying the relaxation parameter $c \in (1, \infty)$. Small $c$ makes the relaxation easier to solve, but its solutions are more likely to deliver the Gromov–Hausdorff distance when $c$ is large.

By default, the method allocates half of the iteration budget to select the best value of $c$ for $(X, Y)$ from $1+10^{-4}, 1+10^{-2},\ldots,1+10^8$, and then spends the remaining half on refining the Gromov–Hausdorff distance using this $c$.

To reveal the value of $c$ selected after the search:

```dgh.upper(X, Y, verbose=1)```

To compute for specific value of $c$ (to avoid spending iterations on the search and/or to test for better accuracy):

```dGH = dgh.upper(X, Y, c=my_c)```


## Contributing
If you found a bug or want to suggest an enhancement, you can create a [GitHub Issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue). Alternatively, you can email vlad.oles (at) proton (dot) me.

## License
dGH is released under the MIT license.

## Research
To cite dGH, you can use the following:
<blockquote>
<p>Oles, V. (2023). Computing the Gromov–Hausdorff distance using first-order methods. <i>arXiv preprint arXiv:2307.13660</i>.</p>
</blockquote>
<pre><code>@article{oles2023computing,
  title={Computing the Gromov--Hausdorff distance using first-order methods},
  author={Oles, Vladyslav},
  journal={arXiv preprint arXiv:2307.13660},
  year={2023}
}
</code></pre>
