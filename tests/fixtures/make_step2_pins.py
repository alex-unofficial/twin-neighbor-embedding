"""Regenerate ``step2_labels.npz`` — the characterisation pins for spec §7 step 2.

Run from the repository root with the project interpreter::

    venv/bin/python tests/fixtures/make_step2_pins.py

The arrays come from ``twin.paper_figures.football_label_arrays`` /
``tabula_label_arrays``, which were lifted **verbatim** out of the figure functions and verified
bit-for-bit against the pre-lift code. Regenerating this file is therefore only ever correct in
a commit that *intends* to change the derived labels or colours and says so (CLAUDE.md, "Commit
granularity"). The point of the fixture is that the restructure in spec §3.4 cannot move these
numbers without a test failing.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

import twin.dataset as legacy
from twin.paper_figures import football_label_arrays, tabula_label_arrays

OUT = Path(__file__).with_name("step2_labels.npz")


def build() -> dict[str, np.ndarray]:
    football = legacy.football_2023()
    fb = football_label_arrays(football)

    tabula = legacy.tabula("liver")
    tb = tabula_label_arrays(tabula, "liver")

    arrays = {
        # --- football_2023: conference / division, and the 3*l+1+s palette -------------
        "football_conference": fb["vertex_labels"],
        "football_division": fb["stand_labels"],
        "football_conference_names": np.array(list(fb["conf_labels"]), dtype=object),
        "football_conference_codes": np.array(list(fb["conf_labels"].values())),
        "football_edge_labels": fb["edge_labels"],
        "football_edge_standing": fb["edge_standing"],
        "football_edge_alpha": fb["edge_alpha"],
        "football_palette": fb["palette"],
        "football_vertex_colors": fb["vertex_colors"],
        "football_edge_colors": fb["edge_colors"],
        # --- tabula-liver: cell type, and the custom_cmap(13, seed=7) palette ----------
        "tabula_cell_type": tb["vertex_labels"],
        "tabula_cell_type_names": np.array(list(tb["label2id"]), dtype=object),
        "tabula_edge_labels": tb["edge_labels"],
        "tabula_palette": tb["palette"],
        "tabula_vertex_colors": tb["vertex_colors"],
        "tabula_edge_colors": tb["edge_colors"],
        "tabula_edge_alpha": tb["edge_alpha"],
        "tabula_edge_marker_size": tb["edge_marker_size"],
        "tabula_alpha_edge_points": tb["alpha_edge_points"],
        # --- digits: the target vector the figures never used but the metrics will -----
        "digits_target": load_digits().target,
    }
    return arrays


if __name__ == "__main__":
    np.savez_compressed(OUT, **build())
    print(f"wrote {OUT}")
