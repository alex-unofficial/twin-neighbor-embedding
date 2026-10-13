"""Persistence and identity for experiment runs (spec §4.6, `DECISIONS.md` (21), (22)).

``twin/runs/`` is **source**; the repository-root ``runs/`` directory is its *output* and is
gitignored. The two share a name on purpose — this package is what reads and writes that one —
and ``.gitignore`` anchors the rule (``/runs/``) so that this directory is not swallowed by it.

:mod:`twin.runs.keys` is the on-disk contract: everything versioned under ``v1-`` lives there,
so that nothing downstream can drift from it.
"""
