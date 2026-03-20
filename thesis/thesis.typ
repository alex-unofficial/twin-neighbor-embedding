#import "conf.typ": styling

#set document(title: [
  Twin Neighbor Embedding: A Pre-Embedding Transform for
  Improved Network Visualization.
])

#show: styling.with(
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

= Introduction

== Problem Description
#include "problem_description.typ"

== State of the Art
#include "sota.typ"
