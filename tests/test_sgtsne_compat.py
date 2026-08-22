"""Spec §1.8 — ``sgtsnepi(silent=True)`` closes fd 2; the wrapper must restore it."""

import os
import sys
import time

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


def test_rss_sampling_across_a_silent_embed_does_not_race_for_the_freed_fd(karate):
    """The sampler must not allocate a descriptor while sgtsnepi is closing one.

    sgtsnepi(silent=True) frees fd 2. A sampler that called open() during the embed would be
    handed that number; preserved_std_fds would then close the sampler's file, and the sampler
    closing it again would take out the restored stderr. RSSSampler holds one fd and preads it,
    so no number is ever in contention. Observed failure, not a hypothetical one.
    """
    from twin.instrumentation import RSSSampler
    from twin.pipelines import graph_embedding

    for _ in range(5):
        with RSSSampler(interval=0.0005):
            emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))
        assert _fd_is_open(1), "stdout was clobbered by the sampler/guard interaction"
        assert _fd_is_open(2), "stderr was clobbered by the sampler/guard interaction"

    # And through the real entry point, which samples by default.
    res = graph_embedding(karate, embedding=emb.SGtSNELayout)
    assert res.peak_mem_mb is not None
    assert _fd_is_open(1) and _fd_is_open(2)
    sys.stdout.write("")
    sys.stderr.write("")


def test_sampler_holds_exactly_one_descriptor():
    """If it allocated per sample, the count would climb."""
    from twin.instrumentation import RSSSampler

    def open_fds():
        return len(os.listdir("/proc/self/fd"))

    before = open_fds()
    with RSSSampler(interval=0.0005) as mem:
        during = open_fds()
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            pass
        still = open_fds()
    after = open_fds()

    assert during - before == 1, "expected exactly one held descriptor"
    assert still == during, "descriptor count grew while sampling"
    assert after == before, "descriptor leaked"
    assert mem.n_samples > 2, "the polling thread should have sampled repeatedly"


@pytest.mark.parametrize("silent", [True, False])
def test_two_consecutive_embeds(karate, silent):
    """Guarding must not break the call itself, in either mode."""
    Gm = gm.VertexMatrix(karate)
    first = emb.SGtSNELayout(silent=silent).embed(Gm)
    second = emb.SGtSNELayout(silent=silent).embed(Gm)
    assert first.shape == second.shape == (2, karate.number_of_nodes())
    assert _fd_is_open(1) and _fd_is_open(2)
