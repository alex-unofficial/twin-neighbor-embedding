#import "commenting.typ": *
#import "macros.typ": *
#import "@preview/run-liners:0.1.0": *
#import "@preview/axiom:0.1.0": *

Consider a graph $G = (V, E)$. 
We must assign each vertex $v in V$ a position $x(v) in RR^d$. 

Denote by $n = |V|$ the number of vertices and by $m = |E|$ the number of edges.
Given some ordering of the vertices $v_1, v_2, dots,  v_n$ the layout $X$
is made of $n$ position vectors $x_1, x_2, dots, x_n$ such that $x_i = x(v_i)$.
From a computational perspective the layout $X in RR^(n times d)$ can be treated as 
a conventional matrix whose rows are the vectors $x_(i)^(T)$.

It is obvious that any single graph has infinitely many layouts, given we have
not stated any constraints on $x(v)$. However, not every layout is of the same
quality. Take, for example, $n$ randomly sampled positions in $RR^d$ and
assign each one to a vertex. This is a valid layout, but it encodes no information
about the underlying network. From a data analysis perspective, it is practically useless.

Therefore we must define what makes a quality layout. 
This is not a simple question, and existing methods adopt different perspectives.

Although layout algorithms are presented in many different forms, a large portion
of the literature can be interpreted through the lens of optimization.
Many 
methods explicitly define an objective or energy function that measures the
quality of a layout and then apply numerical procedures to approximately
minimize it. 
Other methods are not explicitly presented as optimization procedures, 
and may describe the algorithm in isolation --- as analogues of
physical systems, heuristic approaches, or functions of algebraic structures 
associated with the graph --- but can still be understood as implicitly 
optimizing a related criterion.
It is useful in these cases, when possible, to identify the implicit objective
or criteria of a method in order to properly understand its behavior and
results.

Under this perspective, a layout algorithm may be viewed as attempting to
approximate a minimizer of an objective function $scr(L)_G (X)$.
If $X^(*) =  argmin_X scr(L)_G (X)$ then the layout algorithm should attempt 
to approximate $X approx X^(*)$.

Once a layout criterion has been defined, the problem becomes that of computing
a configuration $X$ that approximately minimizes this objective. In practice,
layout methods differ primarily in how this configuration is obtained. Two broad
computational approaches are common.
+ *Directly Computed.* \
  The layout is obtained through a direct transformation of $G$ or of
  an algebraic structure associated with $G$. In such cases the configuration
  $X$ is directly computed as
  $ X = f(G) $
  where $f$ is typically derived from algebraic properties of the graph 
  (for example through matrix factorizations, eigenvalue decompositions 
  or eigenvector computations). 
  When an objective interpretation exists, the resulting configuration may 
  correspond to an exact or approximate solution of the associated optimization problem.

+ *Iterative.* \
  The layout is obtained through an iterative procedure that progressively
  improves a candidate configuration. Given an initial layout $X^((0))$ and an
  update rule $f$, the algorithm generates a sequence
  $ X^((j + 1)) = f(X^((j)), G, j) $
  until a stopping condition is satisfied. Such update rules are often designed
  so that the sequence of layouts reduces the value of the objective $scr(L)_G (X^((j)))$
  at each iteration, or otherwise moves the configuration toward a locally optimal or 
  stable arrangement.

The distinction between these approaches is primarily computational: 
directly computed methods obtain the layout through a single transformation, 
whereas iterative methods refine the layout through repeated updates.

#alex[@yifanhu2015 is a good resource]

=== Force-directed methods
#include "force-based.typ"

=== Spectral Methods
#alex[Laplacian eigenmaps]

=== Stochastic methods
#alex[#sne and #tsne as embedding methods for $k$NN graphs]
