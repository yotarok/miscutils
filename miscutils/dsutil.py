"""Miscellaneous functions around HuggingFace Dataset library."""

from collections.abc import Iterable
import datasets
from typing import TypeAlias, TypeVar
import typing


DatasetLike: TypeAlias = datasets.Dataset | datasets.IterableDataset
DatasetRow: TypeAlias = dict[str, typing.Any]
BatchedDatasetRows: TypeAlias = dict[str, typing.Any]

D = TypeVar("D", bound=DatasetLike)
T = TypeVar("T")


def select_columns(ds: D, columns: Iterable[str]) -> D:
    """Removes columns and make a dataset only contains the specified columns."""
    assert ds.column_names is not None
    to_be_removed = [col for col in ds.column_names if col not in columns]
    return typing.cast(D, ds.remove_columns(to_be_removed))


def _1d_length_filter(x, *, max_len: int, dim: int) -> bool:
    length = 0
    if isinstance(x, list):
        if dim != 0:
            raise ValueError(
                "if dataset contains lists, `dim` of `make_1d_filter_fn` must be 0"
            )
        length = len(x)
    else:
        # TODO: The logic in this branch is not robust, improve.
        length = x.shape[dim]
    return length <= max_len


def apply_1d_length_filter(
    ds: D, column: str, max_len: int, *, dim: int = 0, **kwargs
) -> D:
    return typing.cast(
        D,
        ds.filter(
            _1d_length_filter,
            input_columns=[column],
            fn_kwargs={"max_len": max_len, "dim": dim},
            **kwargs,
        ),
    )


def unbatch(rows: BatchedDatasetRows) -> list[DatasetRow]:
    "Transposes the dictionary of batched arrays to list of dictionary."
    ks = rows.keys()
    return [dict(zip(ks, vs)) for vs in zip(*rows.values())]


def _map_and_reduce_in_batch_fn(
    rows: BatchedDatasetRows, map_fn, reduce_fn
) -> BatchedDatasetRows:
    reduced = reduce_fn([map_fn(row) for row in unbatch(rows)])
    return {
        "_r": [reduced],
    }


def map_and_reduce(
    ds: DatasetLike,
    map_fn: typing.Callable[[DatasetRow], T],
    reduce_fn: typing.Callable[[list[T]], T],
    *,
    num_proc: int | None = None,
    batch_size: int | None = 1000,
) -> T:
    non_iterable_kwargs = {}
    if isinstance(ds, datasets.Dataset):
        # ds is Dataset, use num_proc arg
        non_iterable_kwargs["num_proc"] = num_proc
    else:
        # ds is IterableDataset
        if num_proc is not None:
            raise ValueError(
                "multi-process parallelization not supported for IterableDataset"
            )

    partial = ds.map(
        _map_and_reduce_in_batch_fn,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={"map_fn": map_fn, "reduce_fn": reduce_fn},
        remove_columns=ds.column_names,
        **non_iterable_kwargs,
    )
    return reduce_fn(list(typing.cast(Iterable[T], partial["_r"])))
