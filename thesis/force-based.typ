#import "commenting.typ": *

Force-directed (or force-based) methods are among the most widely used 
approaches for scientific network visualization used in practice. 
The vast majority of algorithms implemented in academic and commercial network
visualization software are based on this principle.
Consequently, force-based methods exhibit substantial variation in
theoretical framing, motivation, and execution.

The central idea is to treat each vertex $v_i$ as a particle located
at position $x_i$ analogous to particles in a physical system.
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

Common families of force-based methods include the spring-electrical model,
as well as the stress and strain models.

#block[*The Spring-Electrical model*]
This model was first introduced by Peter Eades in 1984 @eades1984.
In this first approach, logarithmic strength springs where connected
to every pair of vertices connected by an edge in $G$, while non-adjacent
vertices were subject to a repulsive force inversly proportional to their distance.
Then a physical simulation was used to find a stable configuration starting
from some initial configuration.

Later, Fruchterman and Reingold @FRlayout suggested using attractive spring
forces proportional to the squared distance betweeen adjacent vertices
$ F_a (v_i, v_j) = - (||x_i - x_j||^2)/K (x_i - x_j)/(||x_i - x_j||) quad v_i <-> v_j $
and repulsive electrical forces between all vertices inversly proportional
to their distance in the layout.
$ F_r (v_i, v_j) = (K^2)/(||x_i - x_j||) (x_i - x_j)/(||x_i - x_j||) quad i eq.not j $
where the parameter $K$ is related to the nominal edge length of the final layout.

These forces correspond to an energy function:
$ cal(E)_"FR" (X) = sum_(v_i <-> v_j) (||x_i - x_j||^3)/(3K) + 
  sum_(i eq.not j) K^2 ln(||x_i - x_j||) $

The algorithm for computing the layout consists of calculating for each vertex $v_i$
the force direction vector $f$ as
$ F = sum_(j eq.not i) (F_a (v_i, v_j) + F_r (v_i, v_j)), quad f = F/(||F||) $
according to the previous configuration $X^((j))$ then moving the position in
the direction of the force a distance according to a decreasing step length $h^((j))$
$ x_i^((j + 1)) <- x_i^((j)) + f dot h^((j)) $
until the layout stabilizes to an equilibrium point.
Refinements on the method use an adaptive step length updating scheme 
@bru1995layout @yifanhu2006layout. 

This method usually works well for small graphs, but for larger graphs
it is prone to being trapped in one of the local minima of the energy,
while the algorithm itself must calculate the forces between all $n(n-1)/2$
pairs of points resulting in $cal(O)(n^2)$ computational complexity. 

The time complexity can be improved by using a spatial partioning technique
such as the Barnes & Hut algorithm @barneshut1986, used widely in physics
for simulations of $n$-body problems, which can compute all forces in 
$cal(O)(n log n)$ time with good accuracy. Such an approach is taken
in @tunkelang1999 and @quigley2001.

To effectively overcome the configuration becoming trapped in a local minumum, 
many algorithms #alex(inline: true)[(citations)]
utilize a multilevel (or multiscale) approach, by initially generating
the layout for a coarse approximation $G^((0))$ which captures the
basic connectivity of $G$, then iteratively refining the graph $G^((j))$ 
to include more detail, while using the previously generated layout 
as the initial configuration for the next iteration.
The details of multilevel approaches are outside the scope of this work.

#block[*Metric MDS (Stress)*]

#block[*Classical MDS (Strain)*]

