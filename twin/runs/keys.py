r"""The on-disk contract: case identity, row schemas, atomic shards (`DECISIONS.md` (21), (22)).

Everything a run directory is keyed on lives in this module, so that a change to any of it is a
change to one file — and, if it alters what is written, a deliberate bump of :data:`KEY_VERSION`
that invalidates every existing run directory. Nothing else in the package may build a key id,
a shard schema or a shard path by hand.

⚠️ **Two serialisers, and they are not interchangeable.**

* :func:`canonical_json` is a **digest input** and nothing more. It follows (21) verbatim:
  ``repr(float(v))`` maps ``1`` and ``1.0`` both to ``"1.0"`` *on purpose*, so that
  ``lambda_par=1`` and ``lambda_par=1.0`` resume onto each other. It is lossy by design — decoding
  it would hand back ``lambda_par == "1.0"``, a string, and silently stringify the results
  table.
* :func:`key_to_json` / :func:`decode_key_json` are **type-preserving** and round-trip exactly.
  This is what a shard stores, and what the per-axis parameter columns are materialised from.

⚠️ **`extra` is the whole fixed set, not two names.** Fixed kwargs are deliberately absent from
a :data:`~twin.batching.BatchKey` (``tests/test_batch_key.py::test_a_fixed_kwarg_is_not_in_the_key``),
so two runs at different seeds — or at a fixed ``lambda_par=10`` against a fixed ``lambda_par=1``
— produce *identical* keys. (21) lists ``seed`` and λ as what ``extra`` carries; that closes two
instances of a collision class. :func:`fixed_extra` closes the class: every fixed kwarg of every
axis, mechanically, plus the runner's scalars. (21) specifies ``extra``'s *encoding*, not its
membership, so this is compatible with it rather than a departure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from twin.batching import BatchCase, BatchKey

__all__ = [
    "KEY_VERSION",
    "METRIC_ROW_SCHEMA",
    "DIAGNOSTIC_ROW_SCHEMA",
    "DIGEST_METADATA_KEY",
    "GRAPH_SIDE_ROW_SCHEMA",
    "dataset_identity",
    "params_from_json",
    "canonical_json",
    "case_identity",
    "decode_key_json",
    "fixed_extra",
    "key_digest",
    "key_from_json",
    "key_id",
    "key_slug",
    "key_to_json",
    "read_shard_metadata",
    "shard_is_current",
    "write_shard",
]

#: The encoding version. Bumping it invalidates every existing run directory — change it only
#: deliberately, in a commit that says so.
KEY_VERSION = "v1"

#: Characters a slug may keep; everything else becomes ``-`` ((21)).
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._=,-]")
SLUG_MAX = 80
DIGEST_HEX = 16


# --------------------------------------------------------------------------- #
#                 The canonical form — a digest input, lossy on purpose         #
# --------------------------------------------------------------------------- #


def _canonical_scalar(value: Any) -> str:
    """(21)'s normalisation table, case by case.

    ``bool`` is tested **before** the numeric rule, because ``bool`` is an ``int`` and would
    otherwise digest ``True`` as ``"1.0"``. NumPy scalars are unwrapped first so that a
    ``np.float64(1.0)`` from a sweep built with ``np.linspace`` digests like the literal ``1.0``.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value))
    if value is None:
        return "null"
    return repr(value)


def canonical_json(key: BatchKey, extra: Mapping[str, Any] | None = None) -> str:
    """The digest input of (21): **not** a serialisation to be read back. See the module doc."""
    payload = {
        "axes": [
            [axis, name, [[k, _canonical_scalar(v)] for k, v in sorted(params)]]
            for axis, name, params in key
        ],
        "extra": {k: _canonical_scalar(v) for k, v in sorted((extra or {}).items())},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def key_digest(key: BatchKey, extra: Mapping[str, Any] | None = None) -> str:
    """First 16 hex characters of BLAKE2b over :func:`canonical_json`.

    Not Python's ``hash()``, which is salted per process and would break resumability across
    runs. The digest is taken over the **untruncated** key: collision-freedom rests on it alone.
    """
    return hashlib.blake2b(canonical_json(key, extra).encode()).hexdigest()[:DIGEST_HEX]


def _slug_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value))
    return str(value)


def key_slug(key: BatchKey) -> str:
    """The readable prefix: ``axis=name[p=v,p=v]`` segments joined by ``__``, sanitised, ≤ 80.

    Decoration for a human reading a directory listing, and nothing else — two cases whose slugs
    truncate identically still get different ids, because the digest is over the whole key.
    """
    segments = []
    for axis, name, params in key:
        segment = f"{axis}={name}"
        if params:
            inner = ",".join(f"{k}={_slug_value(v)}" for k, v in sorted(params))
            segment += f"[{inner}]"
        segments.append(segment)
    return _SLUG_UNSAFE.sub("-", "__".join(segments))[:SLUG_MAX]


def key_id(key: BatchKey, **extra: Any) -> str:
    """``"v1-{slug}-{digest}"`` — (21) verbatim. The id a layout file and a shard are named by."""
    return f"{KEY_VERSION}-{key_slug(key)}-{key_digest(key, extra)}"


def fixed_extra(case: Mapping[str, Mapping[str, Any]], **runner_scalars: Any) -> dict[str, Any]:
    """Every fixed kwarg of every axis, namespaced ``axis.param``, plus the runner's scalars.

    Derived mechanically — ``kwargs - dict(key_params)`` per axis — rather than enumerated,
    because an enumerated list is the two names somebody remembered. ``tabula``'s ``organ`` is
    fixed in ``batches/batch.py`` and would otherwise be invisible to the key, so a liver run
    and a lung run would resume onto each other.

    ⚠️ **What this cannot see is a default.** A kwarg left at the factory's or the class's
    default never reaches ``kwargs``, so if a default changes between two runs the ids do not.
    That is the asymmetry `make_batch_key` documents for the key itself, and it is the
    conservative side of it only for *explicit* values: spell a parameter out in the batch spec
    when its identity matters.

    Args:
        case: a :data:`~twin.batching.BatchCase`.
        **runner_scalars: the arguments the runner applies to every case — ``seed``, ``d``,
            ``normalize``, ``common_init`` — which are not batch axes at all.
    """
    extra: dict[str, Any] = {}
    for axis, choice in case.items():
        swept = dict(choice["key_params"])
        for name, value in choice["kwargs"].items():
            if name not in swept:
                extra[f"{axis}.{name}"] = value
    clash = set(extra) & set(runner_scalars)
    if clash:  # pragma: no cover - namespacing makes this unreachable unless a scalar has a dot
        raise ValueError(f"runner scalars collide with axis kwargs: {sorted(clash)}")
    extra.update(runner_scalars)
    return extra


# --------------------------------------------------------------------------- #
#                 The type-preserving form — what a shard stores               #
# --------------------------------------------------------------------------- #


def _encode(value: Any) -> Any:
    """A JSON value that decodes back to exactly ``value``.

    JSON distinguishes ``1`` from ``1.0`` and ``true`` from ``1`` already, so scalars pass
    through. Tuples are tagged, because a batch spec's tuple is a *scalar* and a list a *sweep*
    (`twin/batching.py`'s grammar), so the two must not decode to one another.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError(f"cannot persist a mapping with non-string keys: {value!r}")
        return {"__dict__": {k: _encode(v) for k, v in value.items()}}
    raise TypeError(
        f"cannot persist a {type(value).__name__} in a key ({value!r}). Keys and extras must be "
        "built from scalars, tuples, lists and string-keyed mappings, so that a shard can be "
        "read back without unpickling anything."
    )


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode(v) for v in value["__tuple__"])
        if set(value) == {"__dict__"}:
            return {k: _decode(v) for k, v in value["__dict__"].items()}
        raise ValueError(f"unrecognised tagged value in a key record: {value!r}")
    return value


def key_to_json(key: BatchKey, extra: Mapping[str, Any] | None = None) -> str:
    """Type-preserving JSON for ``(key, extra)``. Round-trips through :func:`decode_key_json`."""
    payload = {
        "version": KEY_VERSION,
        "key": [[axis, name, [[k, _encode(v)] for k, v in params]] for axis, name, params in key],
        "extra": {k: _encode(v) for k, v in sorted((extra or {}).items())},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_key_json(s: str) -> tuple[BatchKey, dict[str, Any]]:
    """``(key, extra)`` exactly as :func:`key_to_json` received them, types and order intact."""
    payload = json.loads(s)
    if payload.get("version") != KEY_VERSION:
        raise ValueError(
            f"key record is version {payload.get('version')!r}, this code reads {KEY_VERSION!r}"
        )
    key = tuple(
        (axis, name, tuple((k, _decode(v)) for k, v in params))
        for axis, name, params in payload["key"]
    )
    return key, {k: _decode(v) for k, v in payload["extra"].items()}


def key_from_json(s: str) -> BatchKey:
    """The :data:`~twin.batching.BatchKey` alone; see :func:`decode_key_json` for the extras."""
    return decode_key_json(s)[0]


def dataset_identity(name: str, params: Mapping[str, Any]) -> dict[str, str]:
    """The two identity columns of a graph-side row: registry name and factory params.

    The params are ``Dataset.params`` — what the factory was actually called with, defaults
    included — so a dataset's rows do not depend on how the batch spec happened to spell it.
    """
    return {
        "graph": name,
        "graph_params": json.dumps(
            {k: _encode(v) for k, v in sorted(params.items())}, sort_keys=True,
            separators=(",", ":"),
        ),
    }


def params_from_json(s: str) -> dict[str, Any]:
    """The inverse of :func:`dataset_identity`'s ``graph_params``, types intact."""
    return {k: _decode(v) for k, v in json.loads(s).items()}


def case_identity(key: BatchKey, extra: Mapping[str, Any]) -> dict[str, str]:
    """The four identity columns every row of a case carries."""
    return {
        "key_id": key_id(key, **extra),
        "slug": key_slug(key),
        "key_json": key_to_json(key, extra),
    }


# --------------------------------------------------------------------------- #
#                                 Row schemas                                  #
# --------------------------------------------------------------------------- #

#: Identity columns, shared by both shard schemas. The three axis-name columns are fixed; the
#: per-axis *parameter* columns are not, because ``key_to_flat_record`` emits
#: ``embedding.linlog`` only for the ``forceatlas2`` cases — per-case shards would each have a
#: different column set, and concatenation would promote to ``object``. They are materialised
#: at concat time from ``key_json``, where the full column union is known.
_IDENTITY_FIELDS = [
    pa.field("key_id", pa.string(), nullable=False),
    pa.field("slug", pa.string(), nullable=False),
    pa.field("key_json", pa.string(), nullable=False),
    pa.field("graph", pa.string()),
    pa.field("representation", pa.string()),
    pa.field("embedding", pa.string()),
]

_METRIC_FIELDS = [
    pa.field("point_set", pa.string()),
    pa.field("label_set", pa.string()),
    pa.field("metric", pa.string(), nullable=False),
    pa.field("variant", pa.string()),
    pa.field("k", pa.int64()),
]

#: One row per (case, point set, label set, metric, variant, k) — the tidy long frame of §3.7.
METRIC_ROW_SCHEMA = pa.schema(
    _IDENTITY_FIELDS
    + _METRIC_FIELDS
    + [
        pa.field("value", pa.float64()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("skip_reason", pa.string()),
        pa.field("n_excluded", pa.int64()),
        pa.field("metric_set_digest", pa.string(), nullable=False),
    ]
)

#: Everything (18) keeps out of the metric frame: counts with their own denominators, and the
#: per-metric cost. Keyed by the same columns, plus the diagnostic's name.
DIAGNOSTIC_ROW_SCHEMA = pa.schema(
    _IDENTITY_FIELDS
    + _METRIC_FIELDS
    + [
        pa.field("diagnostic", pa.string(), nullable=False),
        pa.field("value", pa.float64()),
        pa.field("metric_set_digest", pa.string(), nullable=False),
    ]
)

#: The graph-side table (§5.3 guard 2): Sil\*, Acc\* and their reference levels, keyed by the
#: dataset and nothing else, because they are functions of ``(dataset, label set)`` alone.
#: ``graph_params`` is the factory's parameters in the type-preserving encoding, so that a
#: dataset built twice with different params cannot share a row.
GRAPH_SIDE_ROW_SCHEMA = pa.schema(
    [
        pa.field("graph", pa.string(), nullable=False),
        pa.field("graph_params", pa.string(), nullable=False),
        pa.field("label_set", pa.string()),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("variant", pa.string()),
        pa.field("value", pa.float64()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("skip_reason", pa.string()),
        pa.field("n_excluded", pa.int64()),
        pa.field("metric_set_digest", pa.string(), nullable=False),
    ]
)

#: Where a shard records the metric-set digest it was computed under, in the parquet footer —
#: readable without loading a row, which is what makes the resume check cheap.
DIGEST_METADATA_KEY = b"twin.metric_set_digest"


# --------------------------------------------------------------------------- #
#                                Atomic shards                                 #
# --------------------------------------------------------------------------- #


def write_shard(
    path: str | os.PathLike,
    rows: Iterable[Mapping[str, Any]],
    *,
    schema: pa.Schema,
    metric_set_digest: str,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write ``rows`` to ``path`` atomically, with ``metric_set_digest`` in the footer.

    Written to a temporary file **in the same directory** and moved into place with
    :func:`os.replace`, which is atomic only within one filesystem — a temporary file under
    ``/tmp`` would make it a copy, and a kill mid-copy would leave a partial shard that looks
    complete. Rows are validated against ``schema`` on the way in, so a row with an unknown
    column or a wrongly typed value fails here rather than at concat time.

    ``metadata`` is written to the footer beside the digest, under ``twin.<name>``, and read
    back by :func:`read_shard_metadata` without loading a row.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    unknown = {name for row in rows for name in row} - set(schema.names)
    if unknown:
        raise ValueError(f"rows carry columns the schema does not have: {sorted(unknown)}")
    table = pa.Table.from_pylist(rows, schema=schema)
    # `from_pylist` fills a missing column with nulls whatever the field's nullability says.
    required = [f.name for f in schema if not f.nullable and table[f.name].null_count]
    if required:
        raise ValueError(f"rows leave non-nullable columns empty: {required}")
    footer = {DIGEST_METADATA_KEY: metric_set_digest.encode()}
    footer.update({f"twin.{k}".encode(): v.encode() for k, v in (metadata or {}).items()})
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **footer})

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            pq.write_table(table, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def read_shard_metadata(path: str | os.PathLike) -> dict[str, str] | None:
    """The ``twin.*`` footer entries of a shard, or ``None`` if it is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        metadata = pq.read_schema(path).metadata or {}
    except (OSError, pa.ArrowException):
        return None
    return {
        k.decode()[len("twin."):]: v.decode()
        for k, v in metadata.items()
        if k.startswith(b"twin.")
    }


def shard_is_current(path: str | os.PathLike, *, metric_set_digest: str) -> bool:
    """Whether ``path`` holds a complete shard computed under ``metric_set_digest``.

    False — never an exception — for a missing file, an unreadable or truncated one, and one
    written under a different metric set. A resume treats all three alike: the case's metrics
    are run again. Only the footer is read.
    """
    path = Path(path)
    if not path.is_file():
        return False
    try:
        metadata = pq.read_schema(path).metadata or {}
    except (OSError, pa.ArrowException):
        return False
    return metadata.get(DIGEST_METADATA_KEY) == metric_set_digest.encode()
