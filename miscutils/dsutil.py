from collections.abc import Iterable
import datasets
from typing import TypeVar
import typing


DatasetLike = datasets.Dataset | datasets.IterableDataset


D = TypeVar("D", bound=DatasetLike)


def select_columns(ds: D, columns: Iterable[str]) -> D:
    assert ds.column_names is not None
    to_be_removed = [col for col in ds.column_names if col not in columns]
    return typing.cast(D, ds.remove_columns(to_be_removed))
