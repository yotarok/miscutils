import random
import typing
from typing import Generic

T = typing.TypeVar("T")


class StreamSampler(Generic[T]):
    """Sampling a fixed-size subset from the stream of samples."""

    def __init__(self, sample_size: int):
        self.state: list[T | None] = [None for _ in range(sample_size)]
        self.input_count = 0

    def add(self, sample: T) -> None:
        self.input_count += 1
        vacants = [i for i, v in enumerate(self.state) if v is None]
        if vacants:
            self.state[vacants[0]] = sample
            return

        for i in range(len(self.state)):
            if random.random() < 1.0 / self.input_count:
                self.state[i] = sample
                break

    @property
    def result(self) -> list[T]:
        return [v for v in self.state if v is not None]
