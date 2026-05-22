#import "commenting.typ": *

As discussed in the previous section, a given graph may correspond to
infinitely many layout configurations. However, these configurations are not
equally useful. Although this is obvious at an abstract level, it is difficult
to define concretely what layout quality means in the context of network
visualization.

In scientific network analysis, a layout is useful insofar as it improves our
understanding of the underlying network. In general, this means that the
geometry of the layout should reflect structural relationships present in the
graph. Vertices, edges, communities, paths, weights, and other network features
should be represented in a way that supports visual interpretation.

The difficulty is that any visually interpretable layout exists in a
low-dimensional space, and therefore cannot preserve the full complexity of a
general network. A layout can only emphasize a subset of the network structure.
Moreover, preserving one structural aspect may obscure another, so every layout
method makes trade-offs in what it represents.
#footnote[
  We have already seen an example of such a trade-off in @force-based between
  the Fruchterman-Reingold and LinLog force models: preserving proximity of
  adjacent vertices and uniformity of edge lengths can inhibit community
  separation, and vice-versa. The $r$-PolyLog model makes this trade-off explicit.
]
Consequently, there is no single universal measure of layout quality. Instead,
a layout should be evaluated according to the degree to which it preserves or
reveals specific aspects of the network structure. The purpose of the following
section is therefore to define a set of evaluation criteria and corresponding
metrics, each measuring a distinct property relevant to the goals of this thesis.

These criteria must connect graph-theoretic structure to geometric structure:
we must specify which properties of the network should be preserved, and what
spatial relationships in the layout are taken to represent them.

In this context, the relative geometry of the layout is central to visual
interpretation. In particular, most of the metrics considered below depend on
relative distances between points, since these determine proximity, separation,
crowding, neighborhood structure, and edge length.
Additionally, the absolute coordinate system of a network layout has no intrinsic meaning.
Two coordinate representations that differ only by translation, rotation,
reflection, or uniform scaling are visually equivalent for the purposes of this
thesis.

In practice, two complementary modes of layout evaluation are commonly used.
+ The first is visual examination. A layout is computed for a graph with known
  structural properties, and the result is inspected to determine whether those
  properties are visually apparent. For example, adjacent vertices should often
  appear in close proximity, communities should be visually distinguishable, and
  graph-theoretic relationships should be reflected in the spatial organization of
  the drawing. This approach has the advantage of directly evaluating the layout
  as a visual object, without requiring formal metric definitions. However, it is
  also inherently subjective, since the interpretation of a layout depends on the
  observer's experience, expectations, and visual judgement.
+ The second mode of evaluation is quantitative. One constructs a set of numerical
  measures, or _metrics_, that attempt to quantify specific aspects of structure
  preservation by comparing geometric properties of the layout with structural
  properties of the underlying graph. Such metrics are useful for comparing layout
  algorithms in a reproducible way, since they provide concrete numerical scores.
  However, they should not be treated as definitive measures of layout quality.
  The choice of metric is itself a modeling decision, and different metrics may
  measure different aspects of the same informal criterion. Moreover, a favorable
  metric value does not necessarily imply that a layout is visually or
  analytically useful in practice.
For this reason, the evaluation in this thesis uses both visual examination and
quantitative metrics. Visual results are used to assess whether the structural
features of a graph are interpretable in the drawing, while quantitative metrics
are used to measure specific properties such as neighborhood preservation,
class separability, point crowding, edge geometry, and global distance
preservation. The remainder of this section defines the metrics used in the
evaluation and clarifies which aspect of layout quality each metric is intended
to measure.
