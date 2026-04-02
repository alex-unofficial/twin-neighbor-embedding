#import "commenting.typ": *
#import "macros.typ": *

#include "context.typ"

== Conventional Methods <conventional>
#include "conventional.typ"

=== Force-directed methods <force-based>
#include "force-based.typ"

=== Spectral Methods
#include "spectral.typ"

=== Methods based on Dimensionality Reduction
#include "dimensionality.typ"

== State of the Art <sota>
#alex[
+ Fruchterman-Reingold: NetworkX `spring_layout(method="force")`
+ Yifan Hu: GraphViz `sfdp`
+ ForceAtlas2: `fa2` python package (+ LinLog variant)
+ Kamada Kawai: NetworkX `kamada_kawai_layout`
+ Stress majorization (MDS style): GraphViz `neato`
+ Spectral: NetworkX `spectral_layout`
+ #sgtsnepi: `sgtsnepi` python package
]
