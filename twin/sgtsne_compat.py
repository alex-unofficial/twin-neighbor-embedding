"""Compatibility wrapper around ``sgtsnepi`` (spec §1.8).

The ``silent=True`` path in the ``sgtsnepi`` C extension closes file descriptor 2 and does not
restore it. The computation is unaffected, but every subsequent write to stderr then raises
``OSError: [Errno 9] Bad file descriptor`` — and loguru logs to stderr, so a batch that passes
``progress_fn=logger.info`` dies at the first progress line after the first SG-t-SNE case.

That blocks logging, progress reporting and resumable overnight batches, so every call in this
package goes through :func:`sgtsnepi`, which duplicates the standard descriptors before the call
and restores them afterwards.
"""

import contextlib
import os
import sys

from sgtsnepi import sgtsnepi as _sgtsnepi_raw

__all__ = ["preserved_std_fds", "sgtsnepi"]


@contextlib.contextmanager
def preserved_std_fds(fds=(1, 2)):
    """Restore ``fds`` to whatever they pointed at on entry.

    Guards against a C extension closing or reassigning them. Descriptors that are already
    invalid on entry are left alone — there is nothing to restore them to.
    """
    sys.stdout.flush()
    sys.stderr.flush()

    saved = {}
    for fd in fds:
        try:
            saved[fd] = os.dup(fd)
        except OSError:
            # Already closed before we got here; not ours to repair.
            continue

    try:
        yield
    finally:
        for fd, backup in saved.items():
            try:
                os.dup2(backup, fd)
            finally:
                os.close(backup)


def sgtsnepi(*args, **kwargs):
    """``sgtsnepi.sgtsnepi`` with the standard descriptors restored afterwards."""
    with preserved_std_fds():
        return _sgtsnepi_raw(*args, **kwargs)
