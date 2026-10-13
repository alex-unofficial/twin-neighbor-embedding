"""What produced a run: the git state, the package versions, the host.

Kept apart from the manifest so that a persisted layout can record the SHA that produced *it* —
under the content-addressed store a run's layouts may come from several commits, and one SHA in
a manifest would be a false record of all but one of them.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from twin.config import PROJ_ROOT

__all__ = ["git_state", "host", "now", "package_versions"]

#: The packages whose version can move a number in a results table.
PACKAGES = (
    "numpy", "scipy", "networkx", "scikit-learn", "pandas", "pyarrow", "sgtsnepi",
    "python-igraph", "igraph", "leidenalg", "pydot",
)


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(cwd: Path = PROJ_ROOT) -> dict:
    """``{"sha", "dirty"}`` for the working tree, or ``sha=None`` outside a repository.

    ``dirty`` counts **tracked** changes only: ``runs/`` is ignored, and an untracked scratch
    file does not change what the SHA's code computes.
    """
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def host() -> dict[str, str]:
    return {"hostname": socket.gethostname(), "platform": platform.platform()}


def now() -> str:
    """UTC, ISO 8601, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
