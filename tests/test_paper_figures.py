"""The paper figures still render (CLAUDE.md's merge criterion, half of which was manual).

**Why this file exists.** CLAUDE.md's rule for merging into `experiment-setup` is "the test suite
passes **and** the existing paper figures still reproduce". Until now the second half was
entirely manual — every session ran it by hand — and nothing in `pytest` imported
`paper_figures` except two label-derivation helpers. So a change that broke the plotting layer
passed `pytest` cleanly, and `plots.py` (614 lines) plus `paper_figures.py` (941 lines) had no
automated coverage at all.

**What this is, and is not.** It is a smoke test: each figure function is called and must return
a Matplotlib figure without raising. That catches the failure mode that actually happens here —
a signature change, a renamed dataset, a moved module, a parameter deleted from a layout class —
which is exactly what took `paper_figures` down at steps 2 and 3.

It is **not** a visual regression test. Nothing compares pixels, so a figure that renders an
empty or wrong picture still passes. Real baselines belong at step 10, when §3.6 refactors the
plotting layer: committing baselines now would pin the output of code that is about to be
rewritten. CLAUDE.md's "commit the visual-regression baseline PNGs" describes that future state
— there are none today, and the line has been corrected to say so.

The `Agg` backend is forced at import so this never tries to open a window on a headless run,
and the figures are closed after each test because Matplotlib keeps them alive globally.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow the backend selection
import pytest  # noqa: E402

from twin import paper_figures as pf  # noqa: E402

#: The six that render in seconds. `barabasi_albert` is commented out in the source and is not
#: one of the eight; `tabula_sapiens_teaser` is opt-in below.
FAST_FIGURES = [
    "tree_of_cliques",
    "tree_of_cliques_curves",
    "small_world",
    "small_world_curves",
    "football_figure",
    "football_curves",
]


@pytest.fixture(autouse=True)
def close_figures():
    """Matplotlib holds every figure until closed; six of these would leak across the suite."""
    yield
    plt.close("all")


@pytest.mark.parametrize("name", FAST_FIGURES)
def test_the_fast_paper_figures_render(name):
    """Each returns a figure with at least one axes, and raises nothing on the way."""
    figure = getattr(pf, name)()

    assert isinstance(figure, plt.Figure), f"{name} did not return a Matplotlib figure"
    assert figure.get_axes(), f"{name} returned a figure with no axes"


def test_the_figure_functions_write_nothing_to_the_repository(tmp_path, monkeypatch):
    """The six render into memory; only the commented-out `barabasi_albert` calls `savefig`.

    Pinned because a figure function that started writing would drop files into the working
    tree, and "start a results run only from a clean working tree" (CLAUDE.md) depends on the
    tree being clean for reasons unrelated to figures.
    """
    monkeypatch.chdir(tmp_path)
    pf.small_world()
    assert not list(tmp_path.iterdir())


@pytest.mark.slow
def test_the_tabula_figure_renders():
    """Opt-in: ~3 minutes, and it covers `tabula_sapiens_blockadj` internally.

    Run with ``venv/bin/python -m pytest -m slow``. Excluded by default because a three-minute
    test in a two-minute suite is a test people stop running.
    """
    figure = pf.tabula_sapiens_teaser()
    assert isinstance(figure, plt.Figure)
    assert figure.get_axes()
