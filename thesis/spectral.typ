#import "macros.typ": *
#import "commenting.typ": *

Spectral layout for network visualization was introduced by Kenneth Hall in 1970 @hall1970
and uses the spectral properties of the graph Laplacian to determine vertex positions.

In Hall's original formulation we seek to minimize the energy function
$ cal(E)_"Spectral" (X) = sum_(i ~ j) w_(i j) ||vec(x)_i - vec(x)_j||^2 $ <energy-hall>
where $w_(i j)$ denotes the weight of edge $e_(i j) in E$ and 
$w_(i j) = 0$ if $e_(i j) in.not E$. For undirected graphs $w_(i j) = w_(j i)$.

This objective penalizes large distances between adjacent vertices,
weighted by their edge weights.
To avoid the trivial solution $vec(x)_i = vec(x)_j$ for all vertices, we impose
orthogonality and normalization constraints $X^T X = I$ and $X^T vec(1) = 0$.

The weighted graph Laplacian is defined as
$ L^w = D - W $ <weighted-laplacian>
where $D$ is the diagonal weighted degree matrix defined as $D_(i i) = sum_(j eq.not i) w_(i j)$ 
and $D_(i j) = 0$ for $i eq.not j$, while $W$ is the weighted adjacency matrix 
$W_(i j) = w_(i j)$ when $i eq.not j$ and $W_(i i) = 0$.

Using the laplacian definition in @weighted-laplacian, the energy @energy-hall can be written as
$ cal(E)_"Spectral" (X) = tr(X^T L^w X), quad "subject to" X^T X = I "and" X^T vec(1) = 0 $ <energy-laplacian>

Minimizing @energy-laplacian subject to $X^T X = I$ and $X^T vec(1) = 0$ yields
solutions $X^*$ whose columns are eigenvectors $X^*_k$ of $L^w$.
The energy at the optimum becomes
$ cal(E)_"Spectral" (X^*) = sum_(k=1)^d lambda_k $ 
where $lambda_k$ are the corresponding eigenvalues. 
For a connected graph, the smallest eigenvalue of $L^w$ is $0$ with 
eigenvector $vec(1) slash sqrt(n)$.
The constraint $X^T vec(1) = 0$ excludes this trivial solution,
therefore the optimal layout is obtained by selecting the $d$ 
smallest non-zero eigenvalues and their corresponding eigenvectors.

The spectral layout energy function penalizes large distances between 
adjacent vertices, particularly those connected by strong edge weights,
producing layouts that vary smoothly over the graph. 
Vertices with many or strong edges tend to be placed near each other, 
while weak connections correspond to larger separations. 

However, the spectral layout method often produces less interpretable results 
for many complex real-world graphs. High-degree vertices tend to collapse toward the center,
vertex crowding is common, and the layout may appear stretched along the 
eigenvector directions.
Consequently, spectral layouts are rarely used as final visualizations and 
instead commonly serve as initial configurations for other layout algorithms.

Normally, a full eigen-decomposition requires $BigO(n^3)$ time. 
However, the layout requires computing only the $d$ smallest non-zero eigenvectors,
so iterative eigensolvers can approximate the solution very efficiently
even for very large graphs @koren2002.

