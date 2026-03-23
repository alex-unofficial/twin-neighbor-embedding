#import "config.typ": draft
#import "commenting.typ": *
#import "styling.typ"

#set document(title: [
  Twin Neighbor Embedding: A Pre-Embedding Transform for
  Improved Network Visualization.
])

#let style = if draft {styling.draft} else {styling.thesis}
#show: style.with(
  author: (
    name: [Alexandros Athanasiadis],
    affil: [
      Department of Electrical and Computer Engineering \
      Aristotle University of Thessaloniki
    ],
    supervisor: [Nikos Pitsianis]
  ),
  abstract: include "abstract.typ",
)

= Introduction <intro>
#include "introduction.typ"

= State of the Art <sota>
#include "sota.typ"

= The Twin Neighbor Embedding method <tne>
#alex[The method, how and why it works.]

= Results and Evaluation <results>
#alex[Including visualization results for qualitative evaluation as well as computer metrics.]

= The TWIN software package and GUI <software>
#alex[Brief description of the Software, the parameters and the GUI]

= Conclusion and Limitations <conclusion>
#alex[What the method improves on, where it doesn't, and how it may be improved in the future.]
