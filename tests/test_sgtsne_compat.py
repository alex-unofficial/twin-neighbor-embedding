"""Spec §1.8 — ``sgtsnepi(silent=True)`` closes fd 2; the wrapper must restore it."""

import os
import sys

import networkx as nx
import pytest

import twin.graphmatrix as gm
from twin import embedding as emb
from twin.sgtsne_compat import preserved_std_fds


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def test_preserved_std_fds_restores_a_closed_descriptor():
    with preserved_std_fds():
        os.close(2)
        assert not _fd_is_open(2)
    assert _fd_is_open(2)
    sys.stderr.write("")  # must not raise


def test_silent_sgtsne_leaves_stderr_usable(karate):
    """The regression itself: after a silent embed, logging must still work.

    Without the wrapper this leaves fd 2 closed, and the next loguru call raises EBADF.
    """
    emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))

    assert _fd_is_open(2), "sgtsnepi(silent=True) closed fd 2 and it was not restored"
    sys.stderr.write("")

    from loguru import logger

    logger.info("stderr is usable after a silent SG-t-SNE embed")


def test_silent_sgtsne_still_produces_a_layout(karate):
    Y = emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))
    assert Y.shape == (2, karate.number_of_nodes())
    assert not (Y == 0).all()


@pytest.mark.parametrize("silent", [True, False])
def test_two_consecutive_embeds(karate, silent):
    """Guarding must not break the call itself, in either mode."""
    Gm = gm.VertexMatrix(karate)
    first = emb.SGtSNELayout(silent=silent).embed(Gm)
    second = emb.SGtSNELayout(silent=silent).embed(Gm)
    assert first.shape == second.shape == (2, karate.number_of_nodes())
    assert _fd_is_open(1) and _fd_is_open(2)
