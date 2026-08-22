"""Regression tests for ``sgtsnepi``'s silent mode and its effect on the standard streams.

Spec §1.8 recorded that ``sgtsnepi(silent=True)`` closed file descriptor 2 and did not restore
it, so every later write to stderr raised ``EBADF`` — and loguru logs to stderr, so a batch
passing ``progress_fn=logger.info`` died at the first progress line after the first SG-t-SNE
case. This repository carried a ``twin/sgtsne_compat.py`` wrapper that duplicated and restored
the descriptors.

**Fixed upstream in sgtsnepi 0.7, and the wrapper has been removed.** These tests remain because
the property is one the batch layer depends on: they pin the requirement against a downgrade or
a regression, and they cover both directions — descriptors must survive, *and* silent mode must
still actually be silent.
"""

import os
import sys
import time

import pytest

import twin.graphmatrix as gm
from twin import embedding as emb


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError:
        return False
    return True


def test_installed_sgtsnepi_is_at_least_0_7():
    from importlib.metadata import version

    installed = tuple(int(p) for p in version("sgtsnepi").split(".")[:2])
    assert installed >= (0, 7), (
        f"sgtsnepi {version('sgtsnepi')} predates the fd-2 fix; either upgrade or restore the "
        "descriptor-guard wrapper removed in this commit"
    )


def test_silent_embed_leaves_the_standard_streams_usable(karate):
    """The original failure: stderr unusable after a silent embed, so logging died."""
    emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))

    assert _fd_is_open(1), "sgtsnepi(silent=True) closed fd 1"
    assert _fd_is_open(2), "sgtsnepi(silent=True) closed fd 2"
    sys.stdout.write("")
    sys.stderr.write("")

    from loguru import logger

    logger.info("stderr is usable after a silent SG-t-SNE embed")


def test_silent_mode_is_still_silent(karate):
    """Guards the other direction: a descriptor fix that stopped suppressing output."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(1)
    try:
        os.dup2(write_fd, 1)
        emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(write_fd)

    captured = os.read(read_fd, 1 << 20)
    os.close(read_fd)
    assert captured == b"", f"silent=True wrote {len(captured)} bytes to stdout"


def test_silent_embed_still_produces_a_layout(karate):
    Y = emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))
    assert Y.shape == (2, karate.number_of_nodes())
    assert not (Y == 0).all()


@pytest.mark.parametrize("silent", [True, False])
def test_repeated_embeds(karate, silent):
    Gm = gm.VertexMatrix(karate)
    first = emb.SGtSNELayout(silent=silent).embed(Gm)
    second = emb.SGtSNELayout(silent=silent).embed(Gm)
    assert first.shape == second.shape == (2, karate.number_of_nodes())
    assert _fd_is_open(1) and _fd_is_open(2)


def test_rss_sampling_across_a_silent_embed_is_safe(karate):
    """Integration property: instrumenting an embed must not disturb the standard streams.

    When silent mode freed fd 2, a sampler that opened a file during the call could be handed
    that number, and the subsequent restore/close took stdio out. Nothing frees a std descriptor
    now, and RSSSampler holds one descriptor rather than allocating per sample, so neither half
    of that interaction remains. Cheap enough to keep asserting.
    """
    from twin.instrumentation import RSSSampler
    from twin.pipelines import graph_embedding

    for _ in range(5):
        with RSSSampler(interval=0.0005):
            emb.SGtSNELayout(silent=True).embed(gm.VertexMatrix(karate))
        assert _fd_is_open(1) and _fd_is_open(2)

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
