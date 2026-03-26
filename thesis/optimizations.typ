#import "macros.typ": *
#import "commenting.typ": *

=== Fast force approximation
A recurring limitation with force-directed methods is the scalability
of computing the forces for each vertex. Calculating forces for each
pair of vertices ---a step necessary in all methods described---
has time complexity $BigO(n^2)$, which is impractical for large graphs.

The time complexity can be improved by using a spatial partitioning technique
such as the Barnes-Hut algorithm @barneshut1986, widely used in physics
for $n$-body simulations, which can approximate the forces in 
$BigO(n log n)$ time with good accuracy. Such an approach is taken
in @tunkelang1999 @quigley2001 and @yifanhu2006layout.

=== Multilevel Methods
Another common problem of optimization-based methods is the configuration 
becoming trapped in a local minimum of the energy or objective, resulting
in worse layouts.

To address the problem, various algorithms #alex(inline: true)[(citations)] 
utilize a multilevel (or multiscale) approach by generating a progressively 
coarser series of graphs $G = G^0, G^1, G^2, dots, G^k$, such that $G^(j + 1)$ is a 
coarse approximation of $G^j$ with fewer vertices, while preserving the
basic connectivity structure. 
Then a progressively finer series of layouts 
$X_k, X_(k-1), X_(k-2), dots, X_0 = X$ is generated, where $X_j$
is a layout of graph $G^j$, using $X_j^((0)) = X_(j + 1)$ as its 
initial configuration.  
The core idea is that by having fewer vertices that encode the same
basic connectivity, the energy function $cal(E)^j$ will correspond 
to a coarse approximation of the original $cal(E)^0$, 
while being smoother and therefore easier to traverse without 
becoming trapped in local minima.
At the same time, since $cal(E)^j$ has the same basic
shape as $cal(E)^(j-1)$, its minimum is likely to be close to 
a minimum of the finer energy function. 
By repeating this process in finer and finer steps, the method 
attempts to refine the computed equilibrium point for progressively
finer versions of $G$.
The final layout $X_0$ is an approximate equilibrium configuration 
of $cal(E)^0$, but is less likely to lie in one of its many local 
minima compared to random initialization.

Notably this approach is not limited to any specific model, or to 
network layout in general, but has found wide application in a 
variety of combinatorial optimization problems 
#alex(inline: true)[(more citations)].

#alex[
  We should consider a multi-level approach to #tne as well.
  A refinement strategy like the one described above could produce
  better results, but there is an even simpler idea:
  first generate a traditional vertex layout using #sgtsne and 
  generate the edge layout as defined in the paper, then use the
  combined vertex--edge layout as the initial configuration for the
  #tne step. This might be able to retain some wanted properties
  of the vertex layout while alleviating crowding. Will test soon.
]
