# dGH

Given the distance matrices of metric spaces $X$ and $Y$, estimates the Gromov–Hausdorff distance $$d_\text{GH}(X, Y) = \frac{1}{2}\inf_{f:X\to Y, g:Y\to X} \text{dis}\Bigg(\Big\\{\big(x, f(x)\big): x \in X\Big\\} \cup \Big\\{\big(g(y), y\big): y \in Y\Big\\}\Bigg),$$ where $$\text{dis}(R) = \sup_{(x, y), (x', y') \in R} \big|d_X(x, x') - d_Y(y, y')\big|$$ is the distortion of a relation $R \subseteq X \times Y$.

The distance is estimated from above by minimizing its parametric relaxation whose solutions are guaranteed to deliver $d_\text{GH}(X, Y)$ for a sufficiently large value of the parameter $c>1$. The quadratic relaxation with affine constraints is minimized using conditional gradient descent in $O(n^3)$ time per iteration, where $n = \max\big\\{|X|, |Y|\big\\}$. The retrieved minimum is an upper bound of (and in many cases equals to) the Gromov–Hausdorff distance $d_\text{GH}(X, Y)$.

A detailed description of the relaxation, its optimality guarantees and optimization landscape, and the approach to minimizing it can be found in [Computing the Gromov–Hausdorff distance using first-order methods](https://arxiv.org/pdf/2307.13660.pdf).

## Installation
To install the package from Python Package Index:

```$ pip install dgh```

## Quickstart

Consider $X$ comprised by the vertices of a $1 \times 10$ rectangle and $Y$ — by the vertices of a unit equilateral triangle together with a point that is 10 away from each of them (see illustration).

<p align="center">
    <img src="https://github.com/vlad-oles/dgh/blob/main/illustration.svg" alt="Illustration of the example" width="300"/>
</p>

To create their distance matrices (the $(i,j)$-th entry stores the distance between the $i$-th and $j$-th points):

```
>>> import numpy as np
>>> X = np.array([[0, 1, 10, 10],
...               [0, 0, 10, 10],
...               [0, 0, 0, 1],
...               [0, 0, 0, 0]])
>>> X += X.T
>>> Y = np.array([[0, 1, 1, 10],
...               [0, 0, 1, 10],
...               [0, 0, 0, 10],
...               [0, 0, 0, 0]])
>>> Y += Y.T
```

To compute (an upper bound of) their Gromov–Hausdorff distance $d_\text{GH}(X, Y)$:

```
>>> import dgh
>>> dGH = dgh.upper(X, Y)
>>> dGH
0.5
```

In this case, the distance $d_\text{GH}(X, Y)=\frac{1}{2}$ is computed exactly.

## Iteration budget

By default, the algorithm is allocated 100 iterations of conditional gradient descent. The algorithm restarts from a random point every time after converging to an approximate solution (i.e. a stationary point) until the iteration budget is depleted. Bigger budget generally means longer run time and better accuracy.

To set the iteration budget:

```
>>> dGH = dgh.upper(X, Y, iter_budget=20)
>>> dGH
0.5
```

## Optimal mapping pair

Every solution is a mapping pair $(f:X\to Y, g:Y\to X)$. To access the mapping pair delivering the retrieved minimum:

```
>>> dGH, f, g = dgh.upper(X, Y, return_fg=True)
>>> f
[2, 2, 3, 3]
>>> g
[1, 1, 1, 2]
```

The $i$-th entry in either mapping stores (the index of) the image of its codomain's $i$-th point. For example, here $g(y_3)=x_2$.

## Relaxation parameter $c>1$
Explicitly specifying $c$ can improve the performance of the algorithm. Small $c$ makes the relaxation easier to minimize, but its solutions are more likely to deliver the Gromov–Hausdorff distance when $c$ is large.

By default, the method allocates half of the iteration budget to select the best value of $c$ from $1+10^{-4}, 1+10^{-2},\ldots,1+10^8$, and then spends the remaining half on refining the Gromov–Hausdorff distance using this $c$.
You can specify $c$ explicitly to see if it results in better accuracy and/or to save iterations on the search.

To see the value of $c$ selected after the search (along with the run summary):

```
>>> dgh.upper(X, Y, verbose=1)
iteration budget 100 | c=auto | dGH≥0
spent 49 iterations to choose c=1.0001
proved dGH≤0.5 after 40 restarts
```

To specify $c$ explicitly:

```
>>> dGH = dgh.upper(X, Y, c=1000)
>>> dGH
0.5
```

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
