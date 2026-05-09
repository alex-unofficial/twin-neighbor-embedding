"""
Author: Alexandros Athanasiadis
Date: 2026-05-05
Description: Generic recursive batch specification parser and executor.

This module is domain-agnostic. It parses a declarative batch specification into
concrete cases and executes those cases using a user-provided runner.

It is intended for experiment-style workflows such as:

    graph × representation × embedding

but the parser itself knows nothing about graphs, layouts, metrics, or any
specific application domain.

===============================================================================
Grammar
===============================================================================

batch_input =
    block
    | [block, block, ...]

block =
    group
    | grouped_block

grouped_block =
    {
        "common": group,
        "groups": [block, block, ...],
    }

group =
    {
        axis_name: category_spec,
        ...
    }

category_spec =
    option
    | [option, option, ...]

option =
    "name"
    | ("name", kwargs)

kwargs =
    {
        fixed_param: scalar,
        swept_param: [value_1, value_2, ...],
    }

Only lists denote swept values. Tuples are treated as scalar values, except for
the special option syntax:

    ("name", kwargs)

===============================================================================
Semantics
===============================================================================

A plain group forms a Cartesian product over its axes.

    {
        "graph": ["karate", "football"],
        "embedding": ["spring", "spectral"],
    }

means:

    graph × embedding

A list of blocks forms an additive union.

    [block_1, block_2, ...]

means:

    block_1 + block_2 + ...

A grouped block:

    {
        "common": common_group,
        "groups": [block_1, block_2, ...],
    }

means:

    common_group × (block_1 + block_2 + ...)

===============================================================================
Registries
===============================================================================

Each axis has a registry mapping names to Python objects.

Example:

    registries = {
        "graph": {
            "karate": nx.karate_club_graph,
            "football": load_football_graph,
        },

        "representation": {
            "vertex": gm.VertexMatrix,
            "edge": gm.EdgeMatrix,
            "twin": gm.TwinEmbeddingMatrix,
        },

        "embedding": {
            "spring": emb.SpringLayout,
            "sgtsne": emb.SGTSNELayout,
        },
    }

A batch spec refers to string names. The parser resolves those strings through
the corresponding registry.

===============================================================================
BatchChoice, BatchCase, BatchKey
===============================================================================

A BatchChoice is a concrete resolved option for one axis:

    {
        "axis": "embedding",
        "name": "sgtsne",
        "object": emb.SGTSNELayout,
        "key_params": (("lambda_par", 1.0),),
        "kwargs": {
            "lambda_par": 1.0,
            "silent": True,
        },
    }

A BatchCase is a dictionary from axis name to BatchChoice:

    {
        "graph": {...},
        "representation": {...},
        "embedding": {...},
    }

A BatchKey is a stable result identifier:

    (
        ("graph", "football", ()),
        ("representation", "twin", (("alpha", 0.5), ("k_max", 2))),
        ("embedding", "sgtsne", (("lambda_par", 1.0),)),
    )

Fixed kwargs are deliberately not included in the BatchKey. Swept kwargs are
included.

Therefore:

    BatchCase = executable object
    BatchKey  = result identifier

Do not reconstruct execution from the key. Use the case for execution.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Iterable, Literal, Mapping


BatchChoice = dict[str, Any]
BatchCase = dict[str, BatchChoice]

BatchKeyPart = tuple[str, str, tuple[tuple[str, Any], ...]]
BatchKey = tuple[BatchKeyPart, ...]

DuplicatePolicy = Literal["error", "skip", "overwrite"]


# =============================================================================
# Grammar predicates
# =============================================================================


def is_option_with_kwargs(x: Any) -> bool:
    """
    Return True if x is an option-with-kwargs specification.

    Valid form:

        ("name", {"param": value, ...})

    The tuple is syntax. It is not treated as a sweep container.
    """

    return (
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], str)
        and isinstance(x[1], dict)
    )


def is_grouped_block(x: Any) -> bool:
    """
    Return True if x is a grouped block.

    Valid form:

        {
            "common": group,
            "groups": [block, block, ...],
        }
    """

    return isinstance(x, dict) and "common" in x and "groups" in x


def is_plain_group(x: Any) -> bool:
    """
    Return True if x is a plain terminal group.

    A plain group is a dictionary that does not contain either reserved
    structural key: "common" or "groups".
    """

    return isinstance(x, dict) and "common" not in x and "groups" not in x


def is_sweep_value(x: Any) -> bool:
    """
    Return True if a keyword argument value denotes a sweep.

    By convention, only lists define sweeps.

    Examples
    --------
    Swept:

        "k": [5, 10, 15]

    Fixed:

        "silent": True
        "shape": (100, 2)

    Tuples are deliberately treated as scalar values, not as sweeps.
    """

    return isinstance(x, list)


# =============================================================================
# Normalization and flattening
# =============================================================================


def normalize_category_items(category_spec: Any) -> list[Any]:
    """
    Normalize a category specification into a list of option specifications.

    Accepted input forms
    --------------------

    Bare option:

        "vertex"

    Option with kwargs:

        ("twin", {"alpha": [0.0, 0.5, 1.0]})

    List of options:

        ["vertex", "edge", ("twin", {...})]

    Returns
    -------
    list[Any]
        A list whose elements are either strings or option-with-kwargs tuples.
    """

    if isinstance(category_spec, str):
        return [category_spec]

    if is_option_with_kwargs(category_spec):
        return [category_spec]

    if isinstance(category_spec, list):
        return category_spec

    raise TypeError(
        f"Invalid category specification: {category_spec!r}. "
        "Expected a string, an option-with-kwargs tuple, or a list of options."
    )


def merge_groups(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge inherited axes with local axes.

    Overlapping axes are rejected.

    Rationale
    ---------
    If the parent contains:

        {"graph": ["karate", "lesmis"]}

    and the child contains:

        {"graph": "football"}

    then it is ambiguous whether the child should override the parent, extend
    it, or form a separate product. This module rejects the overlap.
    """

    overlap = set(parent) & set(child)

    if overlap:
        raise ValueError(
            f"Axis specified more than once in nested batch block: {sorted(overlap)}. "
            "Move the axis to one level, or split the block explicitly."
        )

    return {**parent, **child}


def flatten_batch_blocks(
    batch_input: Any,
    inherited: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Flatten recursive batch grammar into terminal flat groups.

    Parameters
    ----------
    batch_input:
        A block or a list of blocks.

    inherited:
        Axes inherited from enclosing "common" blocks.
        Users normally should not pass this manually.

    Returns
    -------
    list[dict[str, Any]]
        A list of flat groups.

    Example
    -------
    Input:

        {
            "common": {"graph": ["karate", "lesmis"]},
            "groups": [
                {"representation": "vertex", "embedding": "spring"},
                {"representation": "edge", "embedding": "sgtsne"},
            ],
        }

    Output:

        [
            {
                "graph": ["karate", "lesmis"],
                "representation": "vertex",
                "embedding": "spring",
            },
            {
                "graph": ["karate", "lesmis"],
                "representation": "edge",
                "embedding": "sgtsne",
            },
        ]
    """

    if inherited is None:
        inherited = {}

    # Additive composition: [block_1, block_2, ...]
    if isinstance(batch_input, list):
        flat_groups: list[dict[str, Any]] = []

        for block in batch_input:
            flat_groups.extend(
                flatten_batch_blocks(block, inherited=inherited)
            )

        return flat_groups

    # Multiplicative inheritance:
    # {"common": group, "groups": [block_1, block_2, ...]}
    if is_grouped_block(batch_input):
        common = batch_input["common"]
        groups = batch_input["groups"]

        if not isinstance(common, dict):
            raise TypeError(
                f"'common' must be a group dictionary, got {common!r}."
            )

        if not isinstance(groups, list):
            raise TypeError(
                f"'groups' must be a list of blocks, got {groups!r}."
            )

        new_inherited = merge_groups(inherited, common)

        flat_groups: list[dict[str, Any]] = []

        for block in groups:
            flat_groups.extend(
                flatten_batch_blocks(block, inherited=new_inherited)
            )

        return flat_groups

    # Terminal group.
    if is_plain_group(batch_input):
        return [merge_groups(inherited, batch_input)]

    raise TypeError(
        f"Invalid batch block: {batch_input!r}. "
        "Expected a group, a list of blocks, or a {'common': ..., 'groups': ...} block."
    )


# =============================================================================
# Axis expansion
# =============================================================================


def split_swept_and_fixed_kwargs(
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """
    Split keyword arguments into swept and fixed parts.

    Swept kwargs are included in:
        - the actual call kwargs;
        - the result key.

    Fixed kwargs are included in:
        - the actual call kwargs only.

    Example
    -------
    Input:

        {
            "lambda_par": [0.01, 0.1],
            "silent": True,
        }

    Output:

        swept = {
            "lambda_par": [0.01, 0.1],
        }

        fixed = {
            "silent": True,
        }
    """

    swept: dict[str, list[Any]] = {}
    fixed: dict[str, Any] = {}

    for key, value in kwargs.items():
        if is_sweep_value(value):
            swept[key] = value
        else:
            fixed[key] = value

    return swept, fixed


def expand_axis_item(
    axis_name: str,
    item: Any,
    registry: Mapping[str, Any],
) -> list[BatchChoice]:
    """
    Expand one option of one axis into concrete choices.

    Parameters
    ----------
    axis_name:
        Name of the axis, for example "graph", "embedding", or "metric".

    item:
        Either a string option or an option-with-kwargs tuple.

    registry:
        Mapping from option names to actual Python objects.

    Returns
    -------
    list[BatchChoice]

    Each BatchChoice has the form:

        {
            "axis": axis_name,
            "name": option_name,
            "object": registered_object,
            "key_params": tuple of swept parameter pairs,
            "kwargs": full concrete kwargs,
        }
    """

    if isinstance(item, str):
        name = item
        raw_kwargs: dict[str, Any] = {}

    elif is_option_with_kwargs(item):
        name, raw_kwargs = item

    else:
        raise TypeError(
            f"Invalid option for axis {axis_name!r}: {item!r}. "
            "Expected a string or an option-with-kwargs tuple."
        )

    try:
        obj = registry[name]
    except KeyError as exc:
        valid_names = ", ".join(repr(k) for k in registry.keys())
        raise KeyError(
            f"Unknown option {name!r} for axis {axis_name!r}. "
            f"Valid options are: {valid_names}."
        ) from exc

    swept, fixed = split_swept_and_fixed_kwargs(raw_kwargs)

    if not swept:
        return [
            {
                "axis": axis_name,
                "name": name,
                "object": obj,
                "key_params": (),
                "kwargs": dict(fixed),
            }
        ]

    swept_names = tuple(swept.keys())
    swept_value_lists = tuple(swept[name] for name in swept_names)

    choices: list[BatchChoice] = []

    for values in product(*swept_value_lists):
        swept_kwargs = dict(zip(swept_names, values))

        choices.append(
            {
                "axis": axis_name,
                "name": name,
                "object": obj,
                "key_params": tuple(zip(swept_names, values)),
                "kwargs": {
                    **fixed,
                    **swept_kwargs,
                },
            }
        )

    return choices


def expand_axis(
    axis_name: str,
    axis_spec: Any,
    registry: Mapping[str, Any],
) -> list[BatchChoice]:
    """
    Expand a complete axis specification.

    Example
    -------
    Axis:

        "embedding": [
            "spring",
            ("sgtsne", {"lambda_par": [0.01, 0.1]}),
        ]

    expands into choices for:

        spring
        sgtsne(lambda_par=0.01)
        sgtsne(lambda_par=0.1)
    """

    items = normalize_category_items(axis_spec)

    choices: list[BatchChoice] = []

    for item in items:
        choices.extend(
            expand_axis_item(axis_name, item, registry)
        )

    return choices


def expand_group(
    group: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
) -> list[BatchCase]:
    """
    Expand one flat group into concrete cases by Cartesian product.

    Parameters
    ----------
    group:
        A flat group mapping axis names to category specifications.

    registries:
        Mapping from axis names to name/object registries.

    Returns
    -------
    list[BatchCase]

    Example
    -------
    Group:

        {
            "graph": ["karate", "lesmis"],
            "embedding": ["spring", "spectral"],
        }

    produces four cases:

        karate × spring
        karate × spectral
        lesmis × spring
        lesmis × spectral
    """

    axis_names = tuple(group.keys())
    expanded_axes: list[list[BatchChoice]] = []

    for axis_name in axis_names:
        if axis_name not in registries:
            raise KeyError(f"No registry provided for axis {axis_name!r}.")

        expanded_axes.append(
            expand_axis(
                axis_name,
                group[axis_name],
                registries[axis_name],
            )
        )

    cases: list[BatchCase] = []

    for choices in product(*expanded_axes):
        cases.append(dict(zip(axis_names, choices)))

    return cases


def expand_batch(
    batch_input: Any,
    registries: Mapping[str, Mapping[str, Any]],
) -> list[BatchCase]:
    """
    Parse and expand a full recursive batch specification.

    This is the central reusable parser function.

    Parameters
    ----------
    batch_input:
        A batch specification using the recursive grammar.

    registries:
        Mapping from axis names to option registries.

    Returns
    -------
    list[BatchCase]
        Concrete executable cases.

    Notes
    -----
    This function is domain-independent. It does not execute anything.
    It only describes what should be executed.
    """

    flat_groups = flatten_batch_blocks(batch_input)

    cases: list[BatchCase] = []

    for group in flat_groups:
        cases.extend(
            expand_group(group, registries)
        )

    return cases


# =============================================================================
# Keys, case views, and execution
# =============================================================================


def make_batch_key(
    case: Mapping[str, BatchChoice],
    *,
    axis_order: Iterable[str] | None = None,
    sort_axes: bool = False,
) -> BatchKey:
    """
    Build a stable key for a concrete batch case.

    Parameters
    ----------
    case:
        A concrete BatchCase.

    axis_order:
        Optional explicit axis order.

        Example:

            axis_order=["graph", "representation", "embedding"]

        If provided, all axes in axis_order must exist in the case.
        Axes not listed in axis_order are appended afterward in their original
        insertion order.

    sort_axes:
        If True, sort axes alphabetically.
        Ignored if axis_order is provided.

    Returns
    -------
    BatchKey

    Key format:

        (
            ("axis_name", "choice_name", key_params),
            ...
        )

    Fixed kwargs are not included in the key.
    Swept kwargs are included in the key.
    """

    if axis_order is not None:
        ordered_axis_names: list[str] = []

        for axis_name in axis_order:
            if axis_name not in case:
                raise KeyError(
                    f"Axis {axis_name!r} from axis_order is not present in the case."
                )

            ordered_axis_names.append(axis_name)

        for axis_name in case.keys():
            if axis_name not in ordered_axis_names:
                ordered_axis_names.append(axis_name)

    elif sort_axes:
        ordered_axis_names = sorted(case.keys())

    else:
        ordered_axis_names = list(case.keys())

    return tuple(
        (
            axis_name,
            case[axis_name]["name"],
            case[axis_name]["key_params"],
        )
        for axis_name in ordered_axis_names
    )


class BatchCaseView:
    """
    Convenience wrapper around a BatchCase.

    This is optional, but it makes domain-specific runners less verbose.

    Example
    -------
    Without BatchCaseView:

        embedding_cls = case["embedding"]["object"]
        embedding_kw = case["embedding"]["kwargs"]

    With BatchCaseView:

        c = BatchCaseView(case)
        embedding_cls = c.obj("embedding")
        embedding_kw = c.kwargs("embedding")
    """

    def __init__(self, case: Mapping[str, BatchChoice]):
        self.case = case

    def has(self, axis: str) -> bool:
        """Return True if the case contains the given axis."""

        return axis in self.case

    def require(self, *axes: str) -> None:
        """
        Raise KeyError if any required axis is missing.
        """

        missing = set(axes) - set(self.case)

        if missing:
            raise KeyError(f"Batch case is missing required axes: {sorted(missing)}")

    def choice(self, axis: str) -> BatchChoice:
        """Return the full choice dictionary for an axis."""

        return self.case[axis]

    def name(self, axis: str) -> str:
        """Return the selected option name for an axis."""

        return self.case[axis]["name"]

    def obj(self, axis: str) -> Any:
        """Return the registered Python object for an axis."""

        return self.case[axis]["object"]

    def kwargs(self, axis: str) -> dict[str, Any]:
        """Return the concrete kwargs for an axis."""

        return self.case[axis]["kwargs"]

    def key_params(self, axis: str) -> tuple[tuple[str, Any], ...]:
        """Return the swept parameters for an axis."""

        return self.case[axis]["key_params"]


def run_batch_cases(
    cases: Iterable[BatchCase],
    runner: Callable[[BatchCase], Any],
    *,
    duplicate_policy: DuplicatePolicy = "error",
    axis_order: Iterable[str] | None = None,
    sort_axes: bool = False,
    progress: bool = False,
    progress_every: int = 1,
    progress_verbose: bool = False,
    progress_fn: Callable[[str], None] = print,
) -> dict[BatchKey, Any]:
    """
    Execute concrete batch cases using a user-provided runner.

    Parameters
    ----------
    cases:
        Concrete cases produced by expand_batch.

    runner:
        Callable that receives one BatchCase and returns a result.

    duplicate_policy:
        How to handle duplicate keys.

        "error":
            Raise ValueError on duplicate key. Safest default.

        "skip":
            Keep first result and skip later duplicates.

        "overwrite":
            Execute duplicate cases and store the latest result.

    axis_order:
        Optional explicit order for generated keys.

    sort_axes:
        If True, sort axes alphabetically in generated keys.
        Ignored if axis_order is supplied.

    progress:
        If True, print a progress report.

    progress_every:
        Print progress every this many completed cases.
        The final case is always reported.

    progress_fn:
        Function used to emit progress messages.
        Defaults to print.

        You can pass a logger, for example:

            progress_fn=logger.info

    Returns
    -------
    dict[BatchKey, Any]
        Mapping from generated batch keys to runner results.
    """

    valid_duplicate_policies = {"error", "skip", "overwrite"}

    if duplicate_policy not in valid_duplicate_policies:
        raise ValueError(
            f"Invalid duplicate_policy={duplicate_policy!r}. "
            f"Expected one of {sorted(valid_duplicate_policies)}."
        )

    if progress_every <= 0:
        raise ValueError("progress_every must be a positive integer.")

    # Materialize cases so we can know the total count.
    # This is acceptable here because expand_batch already returns a list.
    cases = list(cases)
    total_cases = len(cases)

    if progress or progress_verbose:
        progress_fn(f"Executing {total_cases} cases.")

    results: dict[BatchKey, Any] = {}

    for i, case in enumerate(cases, start=1):
        key = make_batch_key(
            case,
            axis_order=axis_order,
            sort_axes=sort_axes,
        )

        if key in results:
            if duplicate_policy == "error":
                raise ValueError(f"Duplicate batch key generated: {key!r}")

            if duplicate_policy == "skip":
                if progress and (i % progress_every == 0 or i == total_cases):
                    percent = 100.0 * i / total_cases if total_cases else 100.0
                    progress_fn(f"{i}/{total_cases} cases ({percent:.1f}%)")
                continue

            if duplicate_policy == "overwrite":
                pass

        if progress_verbose:
            percent = 100.0 * i / total_cases if total_cases else 100.0
            progress_fn(f"case {i}/{total_cases} ({percent:.1f}%). key={key}")
        elif progress and (i % progress_every == 0 or i == total_cases):
            percent = 100.0 * i / total_cases if total_cases else 100.0
            progress_fn(f"case {i}/{total_cases} ({percent:.1f}%)")

        results[key] = runner(case)

    return results


def run_batch(
    batch_input: Any,
    registries: Mapping[str, Mapping[str, Any]],
    runner: Callable[[BatchCase], Any],
    *,
    duplicate_policy: DuplicatePolicy = "error",
    axis_order: Iterable[str] | None = None,
    sort_axes: bool = False,
    progress: bool = False,
    progress_every: int = 1,
    progress_verbose: bool = False,
    progress_fn: Callable[[str], None] = print,
) -> dict[BatchKey, Any]:
    """
    Convenience function: expand a batch specification and execute it.

    Equivalent to:

        cases = expand_batch(batch_input, registries)
        results = run_batch_cases(cases, runner)

    Parameters
    ----------
    batch_input:
        Recursive batch specification.

    registries:
        Axis registries.

    runner:
        Domain-specific runner.

    duplicate_policy:
        Duplicate key policy.

    axis_order:
        Optional explicit key axis order.

    sort_axes:
        Sort axes alphabetically in keys if axis_order is not supplied.

    progress:
        If True, print progress messages.

    progress_every:
        Print progress every this many cases.

    progress_fn:
        Function used to emit progress messages.

    Returns
    -------
    dict[BatchKey, Any]
    """

    cases = expand_batch(batch_input, registries)

    return run_batch_cases(
        cases,
        runner,
        duplicate_policy=duplicate_policy,
        axis_order=axis_order,
        sort_axes=sort_axes,
        progress=progress,
        progress_every=progress_every,
        progress_verbose=progress_verbose,
        progress_fn=progress_fn,
    )


# =============================================================================
# Optional helpers for common runner patterns
# =============================================================================


def resolve_object_choice(choice: BatchChoice) -> Any:
    """
    Resolve a choice whose object may be a factory or a concrete object.

    If choice["object"] is callable:

        return choice["object"](**choice["kwargs"])

    If choice["object"] is not callable:

        require kwargs to be empty, then return the object.

    This is useful for axes like "graph", where registry entries may be either
    precomputed graphs or graph-generating functions.

    Example
    -------
    Registry:

        {
            "karate": nx.karate_club_graph,
            "my_graph": already_built_graph,
        }

    Specification:

        ("erdos_renyi", {"n": [100, 500], "p": [0.01, 0.05]})

    where the registry object is nx.erdos_renyi_graph.
    """

    obj = choice["object"]
    kwargs = choice["kwargs"]

    if callable(obj):
        return obj(**kwargs)

    if kwargs:
        raise ValueError(
            f"Choice {choice['axis']!r}/{choice['name']!r} has kwargs "
            f"{kwargs!r}, but its registered object is not callable."
        )

    return obj


def require_axes(case: Mapping[str, BatchChoice], *axes: str) -> None:
    """
    Raise KeyError if the case does not contain all required axes.

    This is a functional alternative to BatchCaseView.require(...).
    """

    missing = set(axes) - set(case)

    if missing:
        raise KeyError(f"Batch case is missing required axes: {sorted(missing)}")


def make_simple_function_runner(
    function_axis: str,
) -> Callable[[BatchCase], Any]:
    """
    Create a trivial runner for a batch with one callable axis.

    This is useful for simple cases where the batch spec directly selects
    a function and sweeps its kwargs.

    Example
    -------
    registries:

        registries = {
            "function": {
                "f": f,
                "g": g,
            }
        }

    batch_input:

        {
            "function": [
                ("f", {"x": [1, 2, 3], "scale": 10}),
                ("g", {"x": [1, 2, 3]}),
            ]
        }

    execution:

        runner = make_simple_function_runner("function")
        results = run_batch(batch_input, registries, runner)

    Semantics
    ---------
    For each case, this runner calls:

        selected_function(**selected_kwargs)
    """

    def runner(case: BatchCase) -> Any:
        view = BatchCaseView(case)
        view.require(function_axis)

        fn = view.obj(function_axis)
        kwargs = view.kwargs(function_axis)

        if not callable(fn):
            raise TypeError(
                f"Registered object for axis {function_axis!r} and option "
                f"{view.name(function_axis)!r} is not callable."
            )

        return fn(**kwargs)

    return runner


# =============================================================================
# Minimal result inspection and filtering helpers
# =============================================================================


def key_to_dict(key: BatchKey) -> dict[str, dict[str, Any]]:
    """
    Convert a BatchKey into a nested dictionary.

    Input
    -----
        (
            ("graph", "football", ()),
            ("representation", "twin", (("alpha", 0.5), ("k_max", 2))),
            ("embedding", "sgtsne", (("lambda_par", 1.0),)),
        )

    Output
    ------
        {
            "graph": {
                "name": "football",
                "params": {},
            },
            "representation": {
                "name": "twin",
                "params": {
                    "alpha": 0.5,
                    "k_max": 2,
                },
            },
            "embedding": {
                "name": "sgtsne",
                "params": {
                    "lambda_par": 1.0,
                },
            },
        }
    """

    return {
        axis: {
            "name": name,
            "params": dict(params),
        }
        for axis, name, params in key
    }


def key_axis_name(key: BatchKey, axis: str) -> str | None:
    """
    Return the selected option name for a given axis.

    Example
    -------
        key_axis_name(key, "graph") -> "football"

    Returns None if the axis is absent.
    """

    for axis_name, option_name, _params in key:
        if axis_name == axis:
            return option_name

    return None


def key_axis_params(key: BatchKey, axis: str) -> dict[str, Any]:
    """
    Return swept parameters for a given axis.

    Example
    -------
        key_axis_params(key, "embedding")
        -> {"lambda_par": 1.0}

    Returns an empty dictionary if the axis is absent or has no swept params.
    """

    for axis_name, _option_name, params in key:
        if axis_name == axis:
            return dict(params)

    return {}


def key_param_value(
    key: BatchKey,
    axis: str,
    param: str,
    default: Any = None,
) -> Any:
    """
    Return a swept parameter value from a key.

    Example
    -------
        key_param_value(key, "embedding", "lambda_par")
        -> 1.0

    Fixed kwargs do not appear in the key and therefore cannot be queried here.
    """

    return key_axis_params(key, axis).get(param, default)


def key_has_axis(
    key: BatchKey,
    axis: str,
    *,
    name: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> bool:
    """
    Test whether a key contains an axis, optionally with a specific option name
    and swept parameter constraints.

    Examples
    --------
        key_has_axis(key, "graph", name="football")

        key_has_axis(
            key,
            "embedding",
            name="sgtsne",
            params={"lambda_par": 1.0},
        )
    """

    for axis_name, option_name, key_params in key:
        if axis_name != axis:
            continue

        if name is not None and option_name != name:
            return False

        if params is not None:
            key_param_dict = dict(key_params)

            for param_name, expected_value in params.items():
                if key_param_dict.get(param_name) != expected_value:
                    return False

        return True

    return False


def filter_results(
    results: Mapping[BatchKey, Any],
    predicate: Callable[[BatchKey, Any], bool],
) -> dict[BatchKey, Any]:
    """
    Filter a result dictionary using a predicate over (key, value).

    Example
    -------
        football_results = filter_results(
            results,
            lambda key, value: key_has_axis(key, "graph", name="football"),
        )
    """

    return {
        key: value
        for key, value in results.items()
        if predicate(key, value)
    }


def filter_by_axis(
    results: Mapping[BatchKey, Any],
    axis: str,
    *,
    name: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> dict[BatchKey, Any]:
    """
    Filter results by axis option and/or swept parameter values.

    Examples
    --------
    Results where graph == "football":

        filter_by_axis(results, "graph", name="football")

    Results where embedding == "sgtsne":

        filter_by_axis(results, "embedding", name="sgtsne")

    Results where embedding == "sgtsne" and lambda_par == 1.0:

        filter_by_axis(
            results,
            "embedding",
            name="sgtsne",
            params={"lambda_par": 1.0},
        )

    Notes
    -----
    This only filters over information present in the BatchKey.

    Fixed kwargs are not stored in BatchKey and therefore cannot be filtered
    using this helper unless you deliberately make them swept parameters.
    """

    return filter_results(
        results,
        lambda key, value: key_has_axis(
            key,
            axis,
            name=name,
            params=params,
        ),
    )


def filter_by_axes(
    results: Mapping[BatchKey, Any],
    constraints: Mapping[str, Any],
) -> dict[BatchKey, Any]:
    """
    Filter results by multiple axes.

    Constraint forms
    ----------------

    Simple name constraint:

        {
            "graph": "football",
            "embedding": "sgtsne",
        }

    Name + params constraint:

        {
            "graph": "football",
            "embedding": {
                "name": "sgtsne",
                "params": {"lambda_par": 1.0},
            },
        }

    Param-only constraint:

        {
            "representation": {
                "params": {"k_max": 2},
            },
        }

    Example
    -------
        filter_by_axes(
            results,
            {
                "graph": "football",
                "representation": {
                    "name": "twin",
                    "params": {"k_max": 2},
                },
                "embedding": "sgtsne",
            },
        )
    """

    def matches(key: BatchKey, value: Any) -> bool:
        for axis, constraint in constraints.items():
            if isinstance(constraint, str):
                if not key_has_axis(key, axis, name=constraint):
                    return False

            elif isinstance(constraint, dict):
                name = constraint.get("name")
                params = constraint.get("params")

                if not key_has_axis(key, axis, name=name, params=params):
                    return False

            else:
                raise TypeError(
                    f"Invalid constraint for axis {axis!r}: {constraint!r}. "
                    "Expected a string or a dictionary."
                )

        return True

    return filter_results(results, matches)


def key_to_flat_record(
    key: BatchKey,
    *,
    include_params: bool = True,
    param_sep: str = ".",
) -> dict[str, Any]:
    """
    Convert a BatchKey into a flat dictionary.

    Example
    -------
    Input:

        (
            ("graph", "football", ()),
            ("representation", "twin", (("alpha", 0.5), ("k_max", 2))),
            ("embedding", "sgtsne", (("lambda_par", 1.0),)),
        )

    Output:

        {
            "graph": "football",
            "representation": "twin",
            "representation.alpha": 0.5,
            "representation.k_max": 2,
            "embedding": "sgtsne",
            "embedding.lambda_par": 1.0,
        }

    This is useful before building a pandas DataFrame.
    """

    record: dict[str, Any] = {}

    for axis, name, params in key:
        record[axis] = name

        if include_params:
            for param_name, param_value in params:
                record[f"{axis}{param_sep}{param_name}"] = param_value

    return record


def results_to_records(
    results: Mapping[BatchKey, Any],
    *,
    include_result: bool = True,
    result_key: str = "result",
    include_params: bool = True,
    param_sep: str = ".",
) -> list[dict[str, Any]]:
    """
    Convert a result dictionary to a list of flat records.

    Example
    -------
        records = results_to_records(results, include_result=False)

        # Optional:
        # import pandas as pd
        # df = pd.DataFrame(records)

    Parameters
    ----------
    results:
        Mapping from BatchKey to result objects.

    include_result:
        If True, include the result value in each record.

    result_key:
        Field name used for the result value.

    include_params:
        If True, include swept parameters as flattened fields.

    param_sep:
        Separator between axis name and parameter name.

    Returns
    -------
    list[dict[str, Any]]

    Notes
    -----
    If result values are large arrays or dictionaries, you probably do not want
    to put them directly into a DataFrame. In that case use:

        results_to_records(results, include_result=False)

    and keep the original result dictionary for heavy objects.
    """

    records: list[dict[str, Any]] = []

    for key, value in results.items():
        record = key_to_flat_record(
            key,
            include_params=include_params,
            param_sep=param_sep,
        )

        if include_result:
            record[result_key] = value

        records.append(record)

    return records
