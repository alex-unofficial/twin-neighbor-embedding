"""The on-disk contract of `twin/runs/keys.py` — `DECISIONS.md` (21) and (22).

Every test here guards a failure that would be **silent**: a resume serving one configuration's
results to another, a results table quietly stringified, a half-written shard read as complete.
None of them raises on its own, which is why each is asserted directly.

Tests write to ``tmp_path`` and never to the repository's ``runs/``: a test run must not be able
to pollute a real run directory.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from twin.batching import expand_batch, make_batch_key
from twin.runs.keys import (
    DIAGNOSTIC_ROW_SCHEMA,
    KEY_VERSION,
    METRIC_ROW_SCHEMA,
    _canonical_scalar,
    canonical_json,
    case_identity,
    decode_key_json,
    fixed_extra,
    key_digest,
    key_from_json,
    key_id,
    key_slug,
    key_to_json,
    shard_is_current,
    write_shard,
)

#: Trivial factories, as in `test_batch_key.py`: the engine is domain-agnostic, so the real
#: registries would test themselves rather than the key.
REGISTRIES = {
    "graph": {"karate": lambda **kw: kw, "tabula": lambda **kw: kw},
    "embedding": {"sgtsne": lambda **kw: kw, "spring": lambda **kw: kw},
}

KEY = (
    ("graph", "football_2023", ()),
    ("representation", "twin", (("alpha", 0.5), ("k_max", 2))),
    ("embedding", "sgtsne", (("lambda_par", 1.0),)),
)


# --------------------------------------------------------------------------- #
#                   (21)'s normalisation table, case by case                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, encoded",
    [
        (True, "true"),
        (False, "false"),
        (1, "1.0"),
        (1.0, "1.0"),
        (0.1, "0.1"),
        (-3, "-3.0"),
        (None, "null"),
        ("liver", "'liver'"),
        ((1, 2), "(1, 2)"),
        (np.float64(1.0), "1.0"),
        (np.int64(7), "7.0"),
        (np.bool_(True), "true"),
    ],
)
def test_the_normalisation_table(value, encoded):
    assert _canonical_scalar(value) == encoded


def test_bool_is_tested_before_the_numeric_rule():
    """`bool` is an `int`; without the ordering, `True` would digest as `1.0`."""
    assert _canonical_scalar(True) != _canonical_scalar(1)


def test_an_integer_and_its_float_get_the_same_id():
    """`lambda_par=1` and `lambda_par=1.0` are one configuration and must resume onto each other."""
    as_int = (("embedding", "sgtsne", (("lambda_par", 1),)),)
    as_float = (("embedding", "sgtsne", (("lambda_par", 1.0),)),)
    assert key_id(as_int, seed=0) == key_id(as_float, seed=0.0)


def test_the_canonical_form_is_the_one_21_specifies():
    payload = json.loads(canonical_json(KEY, {"seed": 0, "embedding.silent": True}))
    assert payload == {
        "axes": [
            ["graph", "football_2023", []],
            ["representation", "twin", [["alpha", "0.5"], ["k_max", "2.0"]]],
            ["embedding", "sgtsne", [["lambda_par", "1.0"]]],
        ],
        "extra": {"embedding.silent": "true", "seed": "0.0"},
    }
    # sort_keys and the compact separators: no whitespace anywhere in the digest input.
    assert " " not in canonical_json(KEY, {"seed": 0})


def test_the_digest_is_blake2b_truncated_and_stable_across_processes():
    """Pinned, because Python's `hash()` is salted per process and would pass any in-process
    test. A change to this value is a change to every run directory's file names."""
    import hashlib

    expected = hashlib.blake2b(canonical_json(KEY, {"seed": 0}).encode()).hexdigest()[:16]
    assert key_digest(KEY, {"seed": 0}) == expected
    assert key_digest(KEY, {"seed": 0}) == "98241dd534bce00b"


def test_the_id_has_the_21_shape():
    identifier = key_id(KEY, seed=0)
    assert identifier.startswith(f"{KEY_VERSION}-")
    assert identifier == f"v1-{key_slug(KEY)}-{key_digest(KEY, {'seed': 0})}"
    assert key_slug(KEY) == (
        "graph=football_2023__representation=twin-alpha=0.5,k_max=2.0-__"
        "embedding=sgtsne-lambda_par=1.0-"
    )[:80]


def test_two_keys_whose_slugs_truncate_identically_get_different_ids():
    """Collision-freedom rests on the digest, which is over the *untruncated* key."""
    stem = "x" * 100
    one = (("graph", stem, (("p", 1.0),)),)
    other = (("graph", stem, (("p", 2.0),)),)
    assert key_slug(one) == key_slug(other)
    assert len(key_slug(one)) == 80
    assert key_id(one) != key_id(other)


def test_the_slug_keeps_only_safe_characters():
    key = (("graph", "a b/c", (("organ", "liver & lung"),)),)
    assert set(key_slug(key)) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._=,-")


def test_extra_changes_the_id():
    assert key_id(KEY, seed=0) != key_id(KEY, seed=1)
    assert key_id(KEY) != key_id(KEY, seed=0)


# --------------------------------------------------------------------------- #
#                   `extra` is the whole fixed set, mechanically                #
# --------------------------------------------------------------------------- #


def test_two_runs_at_different_seeds_have_identical_keys_and_different_ids():
    """The failure this whole function exists for. The keys *are* equal — that is the batch
    engine's documented contract — so only `extra` can tell the two runs apart."""
    spec = [{"graph": ["karate"], "embedding": [("sgtsne", {"lambda_par": 10.0})]}]
    case = expand_batch(spec, REGISTRIES)[0]
    key = make_batch_key(case)
    assert key == make_batch_key(expand_batch(spec, REGISTRIES)[0])
    assert key_id(key, **fixed_extra(case, seed=0)) != key_id(key, **fixed_extra(case, seed=1))


def test_a_fixed_kwarg_reaches_extra_and_a_swept_one_does_not():
    """`lambda_par` fixed at 10 and fixed at 1 must not share an id; swept, it is in the key."""
    fixed = expand_batch([{"embedding": [("sgtsne", {"lambda_par": 10.0, "d": 2})]}], REGISTRIES)
    assert fixed_extra(fixed[0]) == {"embedding.lambda_par": 10.0, "embedding.d": 2}

    swept = expand_batch(
        [{"embedding": [("sgtsne", {"lambda_par": [1.0, 10.0], "d": 2})]}], REGISTRIES
    )
    assert [fixed_extra(c) for c in swept] == [{"embedding.d": 2}] * 2
    assert len({key_id(make_batch_key(c), **fixed_extra(c)) for c in swept}) == 2


def test_a_fixed_dataset_parameter_reaches_extra():
    """`("tabula", {"organ": "liver"})` is fixed, so it is not in the key — a liver run and a lung
    run would otherwise resume onto each other."""
    liver, lung = (
        expand_batch([{"graph": [("tabula", {"organ": organ})]}], REGISTRIES)[0]
        for organ in ("liver", "lung")
    )
    assert make_batch_key(liver) == make_batch_key(lung)
    assert key_id(make_batch_key(liver), **fixed_extra(liver)) != key_id(
        make_batch_key(lung), **fixed_extra(lung)
    )


def test_runner_scalars_join_the_axis_kwargs():
    case = expand_batch([{"graph": ["karate"]}], REGISTRIES)[0]
    assert fixed_extra(case, seed=3, d=2) == {"seed": 3, "d": 2}


# --------------------------------------------------------------------------- #
#                  The type-preserving form round-trips exactly                 #
# --------------------------------------------------------------------------- #


def test_the_type_preserving_form_round_trips():
    extra = {"seed": 0, "embedding.lambda_par": 1.0, "embedding.silent": True, "graph.organ": "liver"}
    key, decoded = decode_key_json(key_to_json(KEY, extra))
    assert key == KEY
    assert decoded == extra
    # Equality is not enough: 1 == 1.0 == True in Python. The types are the point.
    for name, value in extra.items():
        assert type(decoded[name]) is type(value), name
    assert type(dict(key[2][2])["lambda_par"]) is float
    assert type(dict(key[1][2])["k_max"]) is int
    assert hash(key) == hash(KEY), "a decoded key must still be usable as a dict key"


def test_the_canonical_form_would_have_stringified_the_value():
    """Why there are two serialisers: decoding the digest input loses the type."""
    canonical = json.loads(canonical_json(KEY))
    assert canonical["axes"][2][2] == [["lambda_par", "1.0"]]  # a string
    assert dict(key_from_json(key_to_json(KEY))[2][2])["lambda_par"] == 1.0  # a float


def test_tuples_and_lists_stay_distinct():
    """In the batch grammar a tuple is a scalar and a list a sweep."""
    extra = {"a": (1, 2), "b": [1, 2], "c": {"nested": (0.5,)}}
    assert decode_key_json(key_to_json((), extra))[1] == extra
    assert isinstance(decode_key_json(key_to_json((), extra))[1]["a"], tuple)


def test_numpy_scalars_are_unwrapped():
    _, extra = decode_key_json(key_to_json((), {"x": np.float64(0.5), "n": np.int64(3)}))
    assert extra == {"x": 0.5, "n": 3}
    assert type(extra["n"]) is int


def test_an_unpersistable_value_is_refused():
    with pytest.raises(TypeError, match="cannot persist"):
        key_to_json((), {"fn": len})


def test_a_record_from_another_version_is_refused():
    record = json.loads(key_to_json(KEY))
    record["version"] = "v0"
    with pytest.raises(ValueError, match="version"):
        decode_key_json(json.dumps(record))


# --------------------------------------------------------------------------- #
#                                   Shards                                     #
# --------------------------------------------------------------------------- #


def _row(**overrides):
    row = {
        **case_identity(KEY, {"seed": 0}),
        "graph": "football_2023",
        "representation": "twin",
        "embedding": "sgtsne",
        "point_set": "V",
        "label_set": None,
        "metric": "adjacency_distance_ratio",
        "variant": None,
        "k": None,
        "value": 1.25,
        "status": "ok",
        "skip_reason": None,
        "n_excluded": None,
        "metric_set_digest": "abc",
    }
    row.update(overrides)
    return row


def test_a_shard_round_trips_with_its_schema(tmp_path):
    path = write_shard(
        tmp_path / "shard.parquet",
        [_row(), _row(metric="knn_recall", k=5, n_excluded=0)],
        schema=METRIC_ROW_SCHEMA,
        metric_set_digest="abc",
    )
    table = pq.read_table(path)
    assert table.schema.remove_metadata() == METRIC_ROW_SCHEMA
    assert table.column("k").to_pylist() == [None, 5]
    assert shard_is_current(path, metric_set_digest="abc")


def test_a_shard_whose_digest_no_longer_matches_is_rerun(tmp_path):
    path = write_shard(
        tmp_path / "s.parquet", [_row()], schema=METRIC_ROW_SCHEMA, metric_set_digest="old"
    )
    assert not shard_is_current(path, metric_set_digest="new")


def test_a_half_written_shard_is_not_current(tmp_path):
    """A truncated file at the path — what a non-atomic writer would leave after a kill."""
    good = write_shard(
        tmp_path / "good.parquet", [_row()], schema=METRIC_ROW_SCHEMA, metric_set_digest="abc"
    )
    data = good.read_bytes()
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(data[: len(data) // 2])
    assert not shard_is_current(broken, metric_set_digest="abc")
    assert not shard_is_current(tmp_path / "absent.parquet", metric_set_digest="abc")


def test_a_failed_write_leaves_nothing_behind(tmp_path):
    """The temporary file is in the target's own directory, and is removed on failure."""
    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError, TypeError)):
        write_shard(
            tmp_path / "s.parquet",
            [_row(value="not a number")],
            schema=METRIC_ROW_SCHEMA,
            metric_set_digest="abc",
        )
    assert list(tmp_path.iterdir()) == []


def test_the_write_is_a_rename_within_the_directory(tmp_path, monkeypatch):
    """`os.replace` is atomic only within a filesystem; the temp file must be a sibling."""
    import os

    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append((a, b)), real(a, b)))
    write_shard(tmp_path / "s.parquet", [_row()], schema=METRIC_ROW_SCHEMA, metric_set_digest="x")
    (src, dst), = seen
    assert os.path.dirname(src) == os.path.dirname(dst) == str(tmp_path)


def test_an_unknown_column_is_refused(tmp_path):
    with pytest.raises(ValueError, match="columns the schema does not have"):
        write_shard(
            tmp_path / "s.parquet",
            [_row(**{"embedding.linlog": True})],
            schema=METRIC_ROW_SCHEMA,
            metric_set_digest="x",
        )


def test_a_missing_required_column_is_refused(tmp_path):
    row = _row()
    del row["status"]
    with pytest.raises(ValueError, match="non-nullable"):
        write_shard(tmp_path / "s.parquet", [row], schema=METRIC_ROW_SCHEMA, metric_set_digest="x")


def test_the_two_schemas_share_their_key_columns():
    """The diagnostics frame joins the metric frame on these, so they must agree exactly."""
    shared = [
        "key_id", "slug", "key_json", "graph", "representation", "embedding",
        "point_set", "label_set", "metric", "variant", "k",
    ]
    for name in shared:
        assert METRIC_ROW_SCHEMA.field(name) == DIAGNOSTIC_ROW_SCHEMA.field(name), name


def test_the_shard_schema_has_no_per_axis_parameter_columns():
    """They are ragged across cases (`embedding.linlog` exists only for forceatlas2), so they
    are materialised at concat time — never in a shard."""
    assert not any("." in name for name in METRIC_ROW_SCHEMA.names)
