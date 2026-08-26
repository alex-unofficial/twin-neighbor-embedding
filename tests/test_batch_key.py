"""The batch key — the contract step 7's resumability rests on (`DECISIONS.md` (21)).

**Why this file exists.** `twin/batching.py` is the largest module in the package, 1562 lines,
and until now it was reached by exactly one smoke test (`test_pipelines.py`), which asserts
cardinality and three field values. A mutation audit found that `make_batch_key` could be
changed to emit `(axis, name, ())` — discarding every swept parameter — and the whole suite
stayed green.

That matters because the key is not a convenience. `DECISIONS.md` (21) makes it a **versioned
on-disk contract**: `key_id(key, **extra)` digests the complete key with BLAKE2b, and a run
resumes by asking whether a shard for that id already exists. If a swept parameter never reaches
the key, then every λ in a sweep collides onto one id, and a resume either skips cases that never
ran or overwrites ones that did — silently, and only on the resume path, which is exactly where
it is most expensive to discover.

These tests cover that contract and nothing else. Covering 1562 lines is not the goal, and
CLAUDE.md says the engine is domain-agnostic and not to be rewritten.
"""

import pytest

from twin.batching import (
    expand_batch,
    key_axis_name,
    key_axis_params,
    key_param_value,
    key_to_dict,
    make_batch_key,
)

#: Two axes with trivial factories. The engine is domain-agnostic (CLAUDE.md), so exercising it
#: through the real dataset and embedding registries would test those instead of this.
REGISTRIES = {
    "graph": {"karate": lambda **kw: ("karate", kw), "football": lambda **kw: ("football", kw)},
    "embedding": {"sgtsne": lambda **kw: ("sgtsne", kw), "spring": lambda **kw: ("spring", kw)},
}


@pytest.fixture
def swept_lambda():
    """One axis swept over two λ, with a fixed kwarg alongside it."""
    spec = [
        {
            "graph": ["karate"],
            "embedding": [("sgtsne", {"lambda_par": [1.0, 10.0], "d": 2})],
        }
    ]
    return [make_batch_key(case) for case in expand_batch(spec, REGISTRIES)]


# --------------------------------------------------------------------------- #
#                      Swept parameters reach the key                          #
# --------------------------------------------------------------------------- #


def test_a_swept_parameter_survives_into_the_key(swept_lambda):
    """The mutation that escaped: a key of `(axis, name, ())` passed every existing test."""
    assert [key_param_value(k, "embedding", "lambda_par") for k in swept_lambda] == [1.0, 10.0]


def test_cases_differing_only_in_a_swept_parameter_get_different_keys(swept_lambda):
    """The property resumability actually needs, stated directly.

    Two cases identical but for λ must not collide, or `key_id` digests them to one shard id.
    """
    assert swept_lambda[0] != swept_lambda[1]
    assert len(set(swept_lambda)) == len(swept_lambda)


def test_every_key_in_an_expansion_is_unique():
    """Across a spec that sweeps two axes at once — the shape §4.2's λ sweep produces."""
    spec = [
        {
            "graph": ["karate", "football"],
            "embedding": [("sgtsne", {"lambda_par": [1.0, 10.0]}), "spring"],
        }
    ]
    keys = [make_batch_key(case) for case in expand_batch(spec, REGISTRIES)]
    assert len(keys) == 6  # 2 graphs x (2 lambdas + 1 spring)
    assert len(set(keys)) == 6, "two cases share a key; a resume would drop one of them"


def test_a_fixed_kwarg_is_not_in_the_key(swept_lambda):
    """`d: 2` is fixed, not swept, so it does not distinguish cases and must not enter the key.

    The asymmetry is deliberate and documented in `make_batch_key`: including fixed kwargs would
    change every shard id whenever an unrelated default moved.
    """
    for key in swept_lambda:
        assert "d" not in key_axis_params(key, "embedding")


def test_an_axis_with_no_swept_parameters_carries_an_empty_tuple(swept_lambda):
    for key in swept_lambda:
        assert key_axis_params(key, "graph") == {}
        assert key_axis_name(key, "graph") == "karate"


# --------------------------------------------------------------------------- #
#                          The accessors agree with it                         #
# --------------------------------------------------------------------------- #


def test_key_to_dict_round_trips_the_whole_key(swept_lambda):
    """`key_to_dict` is how the reporting layer will read a key back (§5.1)."""
    key = swept_lambda[1]
    assert key_to_dict(key) == {
        "graph": {"name": "karate", "params": {}},
        "embedding": {"name": "sgtsne", "params": {"lambda_par": 10.0}},
    }


def test_the_accessors_read_what_make_batch_key_wrote(swept_lambda):
    """Three readers of one structure; if they drift, the frame and the shard ids disagree."""
    for key in swept_lambda:
        as_dict = key_to_dict(key)
        for axis in ("graph", "embedding"):
            assert key_axis_name(key, axis) == as_dict[axis]["name"]
            assert key_axis_params(key, axis) == as_dict[axis]["params"]


def test_a_missing_axis_reads_as_absent_rather_than_raising(swept_lambda):
    """A metric row for a case that has no `representation` axis must not blow up the reporter."""
    key = swept_lambda[0]
    assert key_axis_name(key, "representation") is None
    assert key_axis_params(key, "representation") == {}
    assert key_param_value(key, "representation", "alpha", default="none") == "none"


# --------------------------------------------------------------------------- #
#                        The key is stable, and ordered                        #
# --------------------------------------------------------------------------- #


def test_the_key_is_hashable_and_reproducible(swept_lambda):
    """`key_id` digests a canonical serialisation of it, so it has to be both."""
    spec = [
        {
            "graph": ["karate"],
            "embedding": [("sgtsne", {"lambda_par": [1.0, 10.0], "d": 2})],
        }
    ]
    again = [make_batch_key(case) for case in expand_batch(spec, REGISTRIES)]
    assert again == swept_lambda
    assert len({hash(k) for k in swept_lambda}) == 2


def test_axis_order_is_explicit_not_incidental():
    """Two specs with the axes written in a different order must give the same key under
    `sort_axes`, or the same case digests to two shard ids depending on how it was typed."""
    one = {"graph": ["karate"], "embedding": ["spring"]}
    other = {"embedding": ["spring"], "graph": ["karate"]}

    key_one = make_batch_key(expand_batch([one], REGISTRIES)[0], sort_axes=True)
    key_other = make_batch_key(expand_batch([other], REGISTRIES)[0], sort_axes=True)
    assert key_one == key_other

    # Without it, insertion order is preserved — which is the documented default, and the reason
    # a runner that cares about shard identity should pass one of the two ordering options.
    assert make_batch_key(expand_batch([one], REGISTRIES)[0]) != make_batch_key(
        expand_batch([other], REGISTRIES)[0]
    )


def test_axis_order_can_be_stated_explicitly():
    spec = [{"graph": ["karate"], "embedding": ["spring"]}]
    case = expand_batch(spec, REGISTRIES)[0]
    key = make_batch_key(case, axis_order=["embedding", "graph"])
    assert [axis for axis, _name, _params in key] == ["embedding", "graph"]


def test_an_unknown_axis_in_axis_order_is_refused():
    spec = [{"graph": ["karate"], "embedding": ["spring"]}]
    case = expand_batch(spec, REGISTRIES)[0]
    with pytest.raises(KeyError, match="axis_order"):
        make_batch_key(case, axis_order=["representation"])
