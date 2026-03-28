#import "commenting.typ": *
#import "macros.typ": *

Force-directed (or force-based) methods are among the most widely used 
approaches for scientific network visualization. 
The vast majority of algorithms implemented in academic and commercial network
visualization software are based on this principle.
Consequently, force-based methods exhibit substantial variation in
theoretical framing, motivation, and execution.

The central idea is to treat each vertex $v$ as a particle located
at position $vec(x)(v)$ analogous to a particle in a physical system.
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
based on alternative energy formulations such as LinLog and Multidimensional Scaling (MDS).
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
  - sum_(i < j) K^2 log||vec(x)_i - vec(x)_j|| $ <energy-fr>

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

Despite the physical interpretation, the algorithm does not simulate 
particle dynamics according to classical mechanics. 
Instead it behaves as a heuristic gradient-based optimization 
procedure for minimizing @energy-fr.

This method usually works well for small graphs, but for larger graphs
it is prone to becoming trapped in local minima of the energy function @yifanhu2015.
In addition, the algorithm itself must calculate repulsive interactions between all 
$n(n-1)/2$ vertex pairs, resulting in $BigO(n^2)$ computational complexity.

_Fast force approximation._
The time complexity can be improved by using a spatial partitioning technique
such as the Barnes-Hut algorithm @barneshut1986, widely used in physics
for $n$-body simulations, which can approximate the forces in 
$BigO(n log n)$ time with good accuracy. Such an approach is taken
in @tunkelang1999 @quigley2001 and @yifanhu2006layout.

_Multilevel Methods._
To address the problem of the solution becoming trapped in local minima, 
various algorithms #alex(inline: true)[(citations)] 
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

==== LinLog and $r$-PolyLog models 
The Fruchterman-Reingold model tends to produce layouts with small 
and uniform edge lengths and few edge crossings. 
However, it performs poorly at separating clusters, since edges connecting 
different clusters should typically be longer @noack2004.

The LinLog energy model @noack2004 was introduced by Andreas Noack in 2004,
to improve on the cluster separation problem of the Fruchterman-Reingold
force model. Noack takes an energy-centric approach to defining the 
dynamical system of particles that represent the layout $X$.

Specifically the LinLog energy function $cal(E)_"LinLog"$ is defined as
$ cal(E)_"LinLog" (X) = sum_(i ~ j) ||vec(x)_i - vec(x)_j||
  - sum_(i < j) log||vec(x)_i - vec(x)_j|| $ <energy-linlog>
and the layout problem is framed from the explicit perspective of finding
a minimal-energy configuration $X$.

A specific algorithm to find the minima is not described, but
standard optimization methods in conjunction with
multilevel approaches may be used in practice.

LinLog has better separation of clusters with small diameter
compared to Fruchterman-Reingold, but sacrifices edge length
uniformity as a result.

Noack also introduces the generalized $r$-PolyLog class of energy
models with corresponding energy functions
$ cal(E)_(r"-PolyLog") (X) = sum_(i ~ j) 1/r ||vec(x)_i - vec(x)_j||^r
  - sum_(i < j) log||vec(x)_i - vec(x)_j|| $ <energy-polylog>
Where parameter $r >= 1$ describes the model behaviour. Choice $r = 1$
separates clusters while larger values of $r$ increasingly favor
uniform edge lengths.

Notice that $1$-PolyLog is equivalent to LinLog @energy-linlog, while
the $3$-PolyLog energy is the same as Fruchterman-Reingold @energy-fr,
up to a constant scaling factor.

==== The Stress model
A limitation of Spring-Electrical models like those described above is
that they primarily enforce local edge length constraints and do not 
explicitly encode global distance relationships between vertices.
Real-world networks often include quantitative information on their edges 
in the form of weights representing similarity or distance between the vertices. 
While such information could be incorporated by adjusting the strength of 
attractive forces, the Stress model instead adopts a more principled approach
that attempts to match the distances between all vertex pairs.  

Consider a weighted graph $G = (V, E, w)$ where $V$ and $E$ are the vertex
and edge sets, and $w: E -> W$ maps each edge $e$ to a value $w(e) in W$. 
We denote edges by $e_(i j) = (i, j)$ and write $w_(i j) = w(e_(i j))$ when $e_(i j) in E$.

For each vertex pair $(i,j)$ we derive a target distance $d_(i j) in RR^+$. 
If $e_(i j) in E$, the distance is obtained by applying a transformation
$f: W -> RR^+$ that maps each edge weight to a distance, so that $d_(i j) = f(w_(i j))$.
For non-adjacent vertices, $d_(i j)$ is typically defined as the shortest-path
distance between $i$ and $j$ in $G$ when edges are assigned lengths $f(w_(i j))$.

We assume that $G$ is connected, ensuring that $d_(i j)$ is defined
for all vertex pairs. If the graph is disconnected, its connected components
can be identified and the algorithm applied to each component separately.

The choice of transformation $f$ depends on the information represented 
by the weights. If the weights encode distances between vertices, the identity 
mapping $f(w) = w$ suffices.
If instead $w_(i j)$ represents a similarity or probability, a natural choice
is $f(w) = -log(w)$. This transformation preserves the ordering of similarities
and has the useful property that shortest-path distances become additive:
probabilities along a path multiply while their negative logarithms sum.

Once the ideal distances $d_(i j)$ are defined for all vertex pairs,
the goal is to compute a layout that approximates the ideal distances
as closely as possible.

In the stress model, each vertex pair is connected by a spring with nominal
length $d_(i j)$ and stiffness $k_(i j)$. This yields the energy function
$ cal(E)_"Stress" (X) = sum_(i < j) 
  k_(i j) (||vec(x)_i - vec(x)_j|| - d_(i j))^2 $ <energy-stress>
whose minimum corresponds to the layout that optimally matches the ideal distances.
A typical choice is $k_(i j) = 1 slash d_(i j)^2$ which yields the equivalent formulation
$ sum_(i < j) ((||vec(x)_i - vec(x)_j||) / d_(i j) - 1)^2 $
thereby measuring the relative deviation between the actual and ideal edge lengths.

Although the stress model is often grouped with force-directed methods, 
it is more naturally interpreted as an explicit optimization formulation 
derived from Multidimensional Scaling (MDS) @torgerson1952MDS @kruskal1964MDS.
It was introduced to network layout by Kamada and Kawai in 1989 @kamadakawai1989,
who proposed minimizing @energy-stress using Newton's method.
More recently, stress majorization @gansner2004 has become the preferred approach 
due to its robustness and guaranteed monotonic decrease of the stress function.

The Stress model as defined is not easily scalable for large networks. 
Constructing the model requires computing shortest-path distances for all vertex pairs.
Using Johnson's algorithm this step requires $BigO(n^2 log n + n m)$ time
and $BigO(n^2)$ memory to store the computed distances.
The stress-majorization step additionally requires solving dense linear systems
in each iteration, which can be more computationally expensive than the distance 
computation itself.
Consequently, for very large graphs the method becomes both computationally expensive 
and memory intensive @yifanhu2015. Various optimizations and approximations
have therefore been proposed that trade accuracy for reduced time and memory requirements.
#alex(inline: true)[Citations?]

