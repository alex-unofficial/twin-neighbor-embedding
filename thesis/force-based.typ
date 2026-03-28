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

Notice that $1$-PolyLog is equivalent to LinLog @energy-linlog, while
the $3$-PolyLog energy is the same as Fruchterman-Reingold @energy-fr with $K = 1$ 
(which is already subject to the choice of scale). 

A limitation of Spring-Electrical models like those described above is
that they do not explicitly encode edge length in the layout. Real
world networks often include quantitative information associated
with their edges in the form of weights. They might represent 
similarity or distance between the vertices. A modification could
be made to include stronger or weaker attractive forces to 
influence the distance between each edge. However a more
principled approach is taken in the Stress and Strain models
which attempt to match all vertex distances.  

==== The Stress model
Consider a weighted graph $G = (V, E, w)$ where $V, E$ are the vertex
and edge sets, and $w: E -> W$ is a function mapping each edge $e$ to
a value $w(e) in W$. Typical choices for $W$ include $RR$, $RR^+$, $[0,1]$,
$NN$, $CC$, etc.
We denote by $e_(i j) = (i, j)$ and $w_(i j) = w(e_(i j))$ when $e_(i j) in E$.

For each vertex pair $(i,j)$ we seek to find a distance $d_(i j) in RR^+$. 
If $e_(i j) in E$ then we can set $d_(i j) = f(w_(i j))$ where the function 
$f: W -> RR^+$ maps each weight
to a distance, and depends on what the weights represent.
#footnote[
  for example, if the weights represent a distance between vertices
  then a simple choice of $f(q) = q$ may suffice.
  If weights represent similarity between vertices we might chooce $f(q) = (1 - q)/q$, etc.
]
If $e_(i j) in.not E$ then we typically define $d_(i j)$ as the shortest path
distance.
$ d_(i j) = cases(
  f(w_(i j)) "if" e_(i j) in E,
  min_(k ~ i) {f(w_(i k)) + d_(k j)} "if" e_(i j) in.not E,
) $
We assume that $G$ is a connected graph and thus such a distance $d_(i j)$
may be computed for every $i$ and $j$. In the case of disjoint graphs
we can simply identify the connected-subgraphs and perform the algorithm
for each one seperately.

We compute the ideal distance $d_(i j)$ for each vertex pair in $V^2$ then
we seek to generate a layout which matches the ideal distance for each
pair of vertices.

In the stress model, each vertex pair is connected by a spring with nominal
length $d_(i j)$ and constant $k_(i j)$. This imposes an energy function
$ cal(E)_"Stress" = sum_(i eq.not j) 1/2 k_(i j) (||x_i - x_j|| - d_(i j))^2 $ <energy-stress>
whose minimum corresponds to the layout which optimally matches the ideal distances 
according to this model.
A typical choice for $k_(i j) = 2 slash d_(i j)^2$ which results in energy function
$sum_(i eq.not j) (||x_i - x_j|| slash d_(i j) - 1)^2$, thus measures the relative
difference between the actual and ideal edge length.

This model has its roots in Multidimensional Scaling (MDS) @kruskal1964MDS and
was introduced to network layout by Kamada and Kawai in 1989 @kamadakawai1989.
In order to minimize @energy-stress they proposed using Newton's method on 
each position, one vertex at a time. More recently, the technique of Stress
Majorization @gansner2004 became a preferred method to minimize the stress
model energy due to its robustness.

==== The Strain model


