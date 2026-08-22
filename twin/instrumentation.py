"""Runtime and memory instrumentation for the embed step (spec §3.2).

Memory is sampled as process RSS from a background thread rather than tracked with
:mod:`tracemalloc`. ``tracemalloc`` sees only Python-level allocations, and the primary backend
(``sgtsnepi``) is a C extension — so it would report a blank or misleading number in exactly the
place the "TNE embeds n+m points" cost claim needs one. It also roughly doubles allocation cost,
which would corrupt the ``runtime_s`` measurement sitting beside it.

**The sampler does not allocate a file descriptor while sampling.** ``/proc/self/statm`` is
opened once, on entry, and read with :func:`os.pread` thereafter; ``psutil`` is avoided because
it opens and closes ``/proc`` entries per call.

That started as a fix for a specific hazard — ``sgtsnepi(silent=True)`` used to close fd 2, and a
sampler opening a file mid-call could be handed the freed number, after which the descriptor
restore took stdout and stderr down with it. That bug is fixed upstream in ``sgtsnepi`` 0.7 and
the guard it required has been removed. The design is kept anyway: sampling a tight loop should
not churn descriptors, one ``pread`` beats an ``open``/``read``/``close`` per sample, and the
same hazard applies to any C extension that closes descriptors.

What RSS sampling measures, and what it does not:

- It sees allocations made anywhere in the process, including inside C extensions. That is the
  point.
- It is process-wide, so anything else allocating concurrently is attributed to the embed. Run
  batches single-threaded if the number matters.
- Under glibc, freed memory is not always returned to the OS, so the baseline drifts upward
  across a long batch and ``peak_mem_mb`` is a *lower bound* on the embed's own peak demand.
- Sampling can miss a spike shorter than the interval. Samples are always taken at entry and
  exit, so a monotone allocate-then-free pattern is still bracketed.
"""

import os
import threading

__all__ = ["RSSSampler", "rss_supported"]

_STATM = "/proc/self/statm"
_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_supported() -> bool:
    """Whether this host exposes the ``/proc`` interface the sampler needs."""
    return os.path.exists(_STATM)


class RSSSampler:
    """Context manager sampling process RSS, reporting the peak above the entry baseline.

    Holds a single file descriptor on ``/proc/self/statm`` for its whole lifetime and reads it
    with ``pread``, so it allocates no descriptors while sampling. See the module docstring for
    why that matters.

    Attributes:
        baseline_mb: RSS at entry.
        peak_mb: highest RSS observed between entry and exit.
        delta_mb: ``peak_mb - baseline_mb``, clamped at 0. This is what goes in
            ``LayoutResult.peak_mem_mb``.
        n_samples: how many samples were taken, including the entry and exit ones. A value of 2
            means the work finished inside one interval and only the brackets were recorded.

    Example:
        >>> with RSSSampler() as mem:
        ...     block = [0] * 1_000_000
        >>> mem.delta_mb >= 0
        True
    """

    def __init__(self, interval: float = 0.01, enabled: bool = True):
        self.interval = interval
        self.enabled = enabled and rss_supported()
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
        self.n_samples = 0
        self._fd: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def delta_mb(self) -> float:
        return max(self.peak_mb - self.baseline_mb, 0.0)

    def _read_rss_mb(self) -> float:
        # pread against the held descriptor: no open(), so no fd allocation.
        resident = int(os.pread(self._fd, 128, 0).split()[1])
        return resident * _PAGE_SIZE / 1e6

    def _sample(self) -> None:
        try:
            current = self._read_rss_mb()
        except OSError:  # descriptor went away; stop rather than take the process down
            return
        self.peak_mb = max(self.peak_mb, current)
        self.n_samples += 1

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "RSSSampler":
        if not self.enabled:
            return self
        # Opened here, before any guarded call, and held until __exit__.
        self._fd = os.open(_STATM, os.O_RDONLY)
        self.baseline_mb = self.peak_mb = self._read_rss_mb()
        self.n_samples = 1
        self._thread = threading.Thread(target=self._poll, daemon=True, name="rss-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
