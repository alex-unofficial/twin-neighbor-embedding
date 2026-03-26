#import "commenting.typ": *
#import "macros.typ": *

Force-directed (or force-based) methods are among the most widely used 
approaches for scientific network visualization. 
The vast majority of algorithms implemented in academic and commercial network
visualization software are based on this principle.
Consequently, force-based methods exhibit substantial variation in
theoretical framing, motivation, and execution.

The central idea is to treat each vertex $v$ as a particle located
at position $vec(x)(v)$ analogous to particles in a physical system.
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

The general framework described above gives rise to numerous concrete layout models.
In the following sections several representative approaches are presented, 
beginning with the classical spring–electrical model and followed by methods 
based on alternative energy formulations such as LinLog and multidimensional scaling.
For each method we briefly describe the force or energy model and the algorithm 
used to compute the layout.

==== The Spring-Electrical model
This model was first introduced by Peter Eades in 1984 @eades1984.
In this first approach, each edge in $G$ was modeled as a spring
whose attractive force grew logarithmically with distance, 
while non-adjacent vertices were subject to a repulsive force 
inversely proportional to their distance.
A physical simulation was then used to find a stable layout starting
from an initial configuration.

Later, Fruchterman and Reingold @FRlayout suggested a model in which 
attractive spring forces are proportional to the squared distance 
between adjacent vertices, while repulsive electrical forces act between 
all vertex pairs and are inversely proportional to their distance.

Let the unit direction vector from $j$ to $i$ be defined as
$ uvec(r)_(i j) = (vec(x)_i - vec(x)_j)/(||vec(x)_i - vec(x)_j||) $

The attractive force is
$ vec(F)_a (i, j) = - (||vec(x)_i - vec(x)_j||^2)/K med uvec(r)_(i j) quad i ~ j $
and the repulsive force is
$ vec(F)_r (i, j) = (K^2)/(||vec(x)_i - vec(x)_j||) med uvec(r)_(i j) quad i eq.not j $
where the parameter $K$ represents the desired edge length in the final layout.

These forces correspond to the following energy function @noack2004:
$ cal(E)_"FR" (X) = sum_(i ~ j) 1/(3K) ||vec(x)_i - vec(x)_j||^3
  - sum_(i eq.not j) K^2 ln(||vec(x)_i - vec(x)_j||) $ <energy-fr>

The layout is computed iteratively. For each vertex $i$, the normalized 
force direction $uvec(f)_i$ is calculated as:
$ vec(F)_i = sum_(j ~ i) vec(F)_a (i, j) + sum_(j eq.not i) vec(F)_r (i, j), quad
  uvec(f)_i = vec(F)_i/(||vec(F)_i||) $

Using the previous configuration $X^((j))$,
the vertex position is updated by moving in the direction of 
$uvec(f)_i$ with decreasing step length $h^((j))$
$ vec(x)_i^((j + 1)) <- vec(x)_i^((j)) + h^((j)) med vec(f)_i $
until the layout stabilizes at an equilibrium point.
Later refinements introduce adaptive step-length schemes @bru1995layout @yifanhu2006layout. 

Notably, this is not a true physical simulation of particles
according to classical mechanics, but can be interpreted as 
a heuristic gradient-based minimization scheme for @energy-fr.

This method usually works well for small graphs, but for larger graphs
it is prone to becoming trapped in local minima of the energy function @yifanhu2015.
In addition, the algorithm itself must calculate repulsive forces between all 
$n(n-1)/2$ vertex pairs, resulting in $BigO(n^2)$ computational complexity.
Fast force approximation methods used for improving the scalability, 
as well as the multilevel approach for overcoming local minima in energy
are described in @optimizations.

The Fruchterman-Reingold spring-electrical force model tends to generate 
layouts with small and uniform edge lengths and few edge crossings. 
However, it performs poorly at separating clusters, since edges connecting 
different clusters should typically be longer @noack2004.

==== LinLog and $r$-PolyLog models 
The LinLog energy model @noack2004 was introduced by Andreas Noack in 2004,
to improve on the cluster seperation problem of the Fruchterman-Reingold
force model. Noack takes an energy-centric approach to defining the 
dynamical system of particles that represent the layout $X$.

Specifically the LinLog energy function $cal(E)_"LinLog"$ is defined as
$ cal(E)_"LinLog" (X) = sum_(i ~ j) ||vec(x)_i - vec(x)_j||
  - sum_(i eq.not j) ln(||vec(x)_i - vec(x)_j||) $ <energy-linlog>
and the problem is framed from the explicit perspective of finding
a minimal-energy configuration $X$.

A specific algorithm to find the minima is not described, but
standard optimization methods in conjuction with
multilevel approaches may be used in practice.

LinLog has better separation of clusters with small diameter
compared to Fruchterman-Reingold, but sacrifices edge length
uniformity as a result.

Noack also introduces the generalized $r$-PolyLog class of energy
models with corresponding energy functions
$ cal(E)_(r"-PolyLog") = sum_(i ~ j) 1/r ||vec(x)_i - vec(x)_j||^r
  - sum_(i eq.not j) ln(||vec(x)_i - vec(x)_j||) $ <energy-polylog>
Where parameter $r >= 1$ decribes the model behaviour. Choice $r = 1$
separates clusters while $r -> infinity$ enforces uniform edge
lengths, with compromises between the both extremes.

Notice that the $1$-PolyLog energy is equivalent to LinLog @energy-linlog, while
the $3$-PolyLog energy is the same as Fruchterman-Reingold @energy-fr with $K = 1$ 
(which is already subject to the choice of scale). 

==== The Stress model

==== The Strain model

