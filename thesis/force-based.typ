#import "commenting.typ": *

Force-directed (or force-based) methods are among the most widely used 
approaches for scientific network visualization. 
The vast majority of algorithms implemented in academic and commercial network
visualization software are based on this principle.
Consequently, force-based methods exhibit substantial variation in
theoretical framing, motivation, and execution.

The central idea is to treat each vertex $v$ as a particle located
at position $x(v)$ analogous to particles in a physical system.
Forces are then defined between particles corresponding to the graph topology
and relative position.
The specific form of the forces varies between methods, but in general 
they induce an energy function or dynamical system whose equilibrium 
configurations correspond to the layout $X$.

Methods in this category differ primarily in three aspects:

+ _The specific form of the forces and the corresponding energy function._ \
  Different models define interactions between vertices in different ways.
  For example, spring-electrical methods apply attractive forces along 
  edges and repulsive forces between all vertices, while stress and 
  strain models define spring-like forces between all vertex pairs with 
  natural lengths proportional to their geodesic distance in $G$.

+ _The mathematical approach used to obtain the equilibrium configuration._ \
  Equilibrium configurations may be obtained by direct physical simulation of
  the dynamical particle system, wherein each particle's position is iteratively
  updated according to the forces acting upon it, or by numerical
  methods that compute equilibrium states of the system,
  for example by solving the corresponding equilibrium equations or
  by minimizing the associated energy function. 

+ _The computational algorithms used to calculate the resulting layout._ \
  Among the many force-based layout methods that exist, many differ primarily
  in the implementation strategies, particularly in techniques used to improve 
  scalability and efficiency.
  This is especially important for large networks, where naive implementations
  tend to become impractically slow and memory inefficient. 
  Common approaches include approximation techniques, multilevel methods, and data
  structures that accelerate force computation.

*The Spring-Electrical model*
This model was first introduced by Peter Eades in 1984 @eades1984.
In this first approach, every pair of vertices that form an edge in $G$ was
connected by springs whose attractive force grows logarithmically with distance, 
while non-adjacent vertices were subject to a repulsive force inversely proportional 
to their distance in the layout.
Then a physical simulation was used to find a stable configuration starting
from some initial configuration.

Later, Fruchterman and Reingold @FRlayout suggested using attractive spring
forces proportional to the squared distance between adjacent vertices
and repulsive electrical forces between all vertices inversely proportional
to their distance in the layout.

Using the unit direction vector $hat(r)_(i j) = (x_i - x_j)/(||x_i - x_j||)$,
the attractive force is
$ F_a (i, j) = - (||x_i - x_j||^2)/K med hat(r)_(i j) quad i ~ j $
and the repulsive force is
$ F_r (i, j) = (K^2)/(||x_i - x_j||) med hat(r)_(i j) quad i eq.not j $
where the parameter $K$ is related to the nominal edge length of the final layout.

These forces correspond to an energy function:
$ cal(E)_"FR" (X) = sum_(i ~ j) (||x_i - x_j||^3)/(3K)
  - sum_(i eq.not j) K^2 ln(||x_i - x_j||) $

The algorithm for computing the layout consists of calculating for each vertex $v_i$
the normalized force direction $f_i$ as
$ F_i = sum_(j ~ i) F_a (i, j) + sum_(j eq.not i) F_r (i, j), quad
  f_i = F_i/(||F_i||) $
according to the previous configuration $X^((j))$ then moving the position in
the direction of the force a distance according to a decreasing step length $h^((j))$
$ x_i^((j + 1)) <- x_i^((j)) + f_i dot h^((j)) $
until the layout stabilizes to an equilibrium point.
Refinements on the method use an adaptive step length updating scheme 
@bru1995layout @yifanhu2006layout. 

This method usually works well for small graphs, but for larger graphs
it is prone to being trapped in one of the local minima of the energy,
while the algorithm itself must calculate the forces between all $n(n-1)/2$
vertex pairs, resulting in $cal(O)(n^2)$ computational complexity. 

The time complexity can be improved by using a spatial partitioning technique
such as the Barnes-Hut algorithm @barneshut1986, used widely in physics
for simulations of $n$-body problems, which can compute all forces in 
$cal(O)(n log n)$ time with good accuracy. Such an approach is taken
in @tunkelang1999 and @quigley2001.

To effectively overcome the configuration becoming trapped in a local minimum, 
various algorithms #alex(inline: true)[(citations)] utilize a multilevel 
(or multiscale) approach, by generating a progressively coarser series of graphs 
$G = G^0, G^1, G^2, dots, G^k$, such that $G^(j + 1)$ is a 
coarse approximation of $G^j$, with fewer vertices, while retaining its 
basic connectivity.  Then a progressively finer series of layouts 
$X^k, X^(k-1), X^(k-2), dots, X^0 = X$ is generated, where $X^j$
is a layout of graph $G^j$, using $X^(j + 1)$ as its initial configuration.
The core idea is that by having fewer vertices that encode the same
basic connectivity, the energy function will correspond to a coarse
approximation of the original, while being smoother and therefore
easier to traverse without becoming trapped in local minima,
at the same time having the same general shape of the original 
such that the computed minimum of the coarse energy function 
is likely close to a minimum of the finer energy function. 
By repeating this process in finer and finer steps the method 
attempts to refine the computed equilibrium point to finer versions
of the graph. 
The final layout $X^0$ is an approximate equilibrium configuration 
of the energy function of $G$, only less likely to be in one of the 
many other local minima that exist in the original energy function.
Notably this approach is not limited to the spring-electrical
model, to force-directed methods, or to network layout in general, 
but has found wide application in many combinatorial optimization
problems #alex(inline: true)[(more citations)].

*The Stress model*

*Classical MDS (Strain)*

