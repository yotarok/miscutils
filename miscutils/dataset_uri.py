import dataclasses
from typing import Callable

import aiohttp
import datasets
from loguru import logger


@dataclasses.dataclass
class DatasetUri:
    """Utility class for specifying a dataset from, e.g. command-line argument.

    A dataset URI has the following form.

     - full: "[SOURCE]:[NAME]:[SPLIT]"
     - w/o source: "[NAME]:[SPLIT]"
     - w/o split: "[NAME]"

    Interpretation of NAME and SPLIT is depending on the SOURCE.

    The primary SOURCE is Hugging Face Hub ("hfhub").
    """

    source: str | None
    name: str
    split: str | None

    def load_dataset(self) -> datasets.Dataset:
        loader = (
            _DATA_SOURCES[""] if self.source is None else _DATA_SOURCES[self.source]
        )
        return loader(self)

    @classmethod
    def from_string(cls, s: str) -> "DatasetUri":
        components = s.split(":", maxsplit=2)

        source = None
        name = None
        split = None
        if len(components) == 3:
            source, name, split = components
        elif len(components) == 2:
            name, split = components
        elif len(components) == 1:
            (name,) = components
        else:
            raise ValueError("`DatasetUri.from_string` is called with an empty string")

        return cls(source, name, split)


def load_dataset_from_hugging_face_hub(uri: DatasetUri) -> datasets.Dataset:
    if "@" in uri.name:
        name, data_dir = uri.name.rsplit("@", maxsplit=1)
    else:
        name = uri.name
        data_dir = None

    logger.info(
        f"Loading dataset [{name}:{uri.split}] from hfhub. (data_dir={data_dir})"
    )
    dataset = datasets.load_dataset(
        name,
        data_dir=data_dir,
        split=uri.split,
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )

    return dataset


DataSource = Callable[[DatasetUri], datasets.Dataset]
_DATA_SOURCES: dict[str, DataSource] = {
    "": load_dataset_from_hugging_face_hub,
    "hfhub": load_dataset_from_hugging_face_hub,
}
