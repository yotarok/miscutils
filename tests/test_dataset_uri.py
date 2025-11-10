import unittest.mock
import inspect
from typing import Any, Callable

import datasets
from miscutils import dataset_uri


def _fake_dataset():
    def gen():
        for n in range(10):
            yield {"n": n}

    return datasets.Dataset.from_generator(gen)


def _extract_kwargs(
    mock: unittest.mock.Mock, target_fn: Callable[..., Any]
) -> dict[str, Any]:
    args, kwargs = mock.call_args_list[-1]

    sig = inspect.signature(target_fn)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def test_dataset_uri():
    mock_load_dataset = unittest.mock.MagicMock(return_value=_fake_dataset())
    with unittest.mock.patch("datasets.load_dataset", new=mock_load_dataset):
        dsid = dataset_uri.DatasetUri.from_string("karpathy/tiny_shakespeare")
        dsid.load_dataset()

    assert len(mock_load_dataset.call_args_list) == 1
    call_kwargs = _extract_kwargs(mock_load_dataset, datasets.load_dataset)

    assert call_kwargs["path"] == "karpathy/tiny_shakespeare"
    assert call_kwargs["data_dir"] is None
    assert call_kwargs["split"] is None

    # Again, with scheme specifier. this must result in the same behavior
    mock_load_dataset = unittest.mock.MagicMock(return_value=_fake_dataset())
    with unittest.mock.patch("datasets.load_dataset", new=mock_load_dataset):
        dsid = dataset_uri.DatasetUri.from_string("karpathy/tiny_shakespeare")
        dsid.load_dataset()

    assert len(mock_load_dataset.call_args_list) == 1
    call_kwargs = _extract_kwargs(mock_load_dataset, datasets.load_dataset)

    assert call_kwargs["path"] == "karpathy/tiny_shakespeare"
    assert call_kwargs["data_dir"] is None
    assert call_kwargs["split"] is None


def test_dataset_uri_split():
    mock_load_dataset = unittest.mock.MagicMock(return_value=_fake_dataset())
    with unittest.mock.patch("datasets.load_dataset", new=mock_load_dataset):
        dsid = dataset_uri.DatasetUri.from_string("karpathy/tiny_shakespeare:train")
        dsid.load_dataset()

    # TODO: currently "train[0:20]" split is not supported when scheme is not specified.

    assert len(mock_load_dataset.call_args_list) == 1
    call_kwargs = _extract_kwargs(mock_load_dataset, datasets.load_dataset)

    assert call_kwargs["path"] == "karpathy/tiny_shakespeare"
    assert call_kwargs["data_dir"] is None
    assert call_kwargs["split"] == "train"


def test_dataset_uri_data_dir():
    mock_load_dataset = unittest.mock.MagicMock(return_value=_fake_dataset())
    with unittest.mock.patch("datasets.load_dataset", new=mock_load_dataset):
        dsid = dataset_uri.DatasetUri.from_string(
            "karpathy/tiny_shakespeare@/home/dataset:train"
        )
        dsid.load_dataset()

    # TODO: currently "train[0:20]" split is not supported when scheme is not specified.

    assert len(mock_load_dataset.call_args_list) == 1
    call_kwargs = _extract_kwargs(mock_load_dataset, datasets.load_dataset)

    assert call_kwargs["path"] == "karpathy/tiny_shakespeare"
    assert call_kwargs["data_dir"] == "/home/dataset"
    assert call_kwargs["split"] == "train"
