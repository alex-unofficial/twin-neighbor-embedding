"""Deprecated — the ground-truth labels now live with their datasets (spec §3.4).

This module survives only so that `paper_figures.py` keeps working until its label derivations
move to `Dataset` in the same step. It is deleted then; do not add to it.
"""

from sklearn.datasets import load_digits

from twin.datasets.football_metadata import CONFERENCE, CONFERENCE_STANDING

football_2023 = {"conference": CONFERENCE, "conference_standing": CONFERENCE_STANDING}

_digits = load_digits()
mnist = {"digit": _digits.target, "image": _digits.images}
