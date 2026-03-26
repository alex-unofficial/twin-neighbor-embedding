#import "commenting.typ": *
#import "macros.typ": *
#import "@preview/run-liners:0.1.0": *
#import "@preview/axiom:0.1.0": *

Consider a graph $G = (V, E)$. 
We must assign to each vertex $v in V$ a position $vec(x)(v) in RR^d$. 

Denote by $n = |V|$ the number of vertices and by $m = |E|$ the number of edges.
Given some ordering of the vertices $1, 2, dots, n$ the layout $X$
is made of $n$ position vectors $vec(x)_1, vec(x)_2, dots, vec(x)_n$ such that 
$vec(x)_i = vec(x)(i)$ is the position of vertex $i$.
From a computational perspective the layout $X in RR^(n times d)$ can be treated as 
a conventional matrix whose rows are the vectors $vec(x)_i^T$.

It is obvious that any single graph has infinitely many layouts, given we have
not stated any constraints on $vec(x)(v)$. However, not every layout is of the same
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

In the following sections we briefly review several widely used classes 
of layout algorithms, focusing on representative methods and the 
principles that underlie their formulations.

#alex[@yifanhu2015 is a good resource]

=== Force-directed methods <force-based>
#include "force-based.typ"

=== Spectral Methods
#alex[Laplacian eigenmaps]

=== Stochastic methods
#alex[#sne and #tsne as embedding methods for $k$NN graphs]
