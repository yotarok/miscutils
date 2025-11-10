import datasets
import numpy as np

from miscutils import dsutil


def test_unbatch():
    test_in = {
        "x": np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ]
        ),
        "y": np.random.rand(3, 2, 2),
    }
    test_out = dsutil.unbatch(test_in)
    assert test_out[0]["x"].tolist() == [1, 2]
    assert test_out[0]["y"].tolist() == test_in["y"][0].tolist()
    assert test_out[1]["x"].tolist() == [3, 4]
    assert test_out[1]["y"].tolist() == test_in["y"][1].tolist()
    assert test_out[2]["x"].tolist() == [5, 6]
    assert test_out[2]["y"].tolist() == test_in["y"][2].tolist()


def test_map_and_reduce():
    def gen():
        x0 = 0
        x1 = 1
        for _ in range(10):
            y = x0 + x1
            x0, x1 = x1, y
            yield {"y": y, "ratio": y / x0}

    ds = datasets.Dataset.from_generator(gen)
    assert isinstance(ds, datasets.Dataset)

    sum_y = dsutil.map_and_reduce(ds, lambda row: row["y"], sum)
    max_ratio = dsutil.map_and_reduce(ds, lambda row: row["ratio"], max)

    assert sum_y == 231
    assert max_ratio == 2.0


def test_make_1d_length_filter_fn():
    lengths = [2, 3, 5, 7, 5, 3, 7, 2, 1, 6]

    def gen():
        for length in lengths:
            yield {
                "values": np.random.uniform(size=(length,)).tolist(),
                "length": length,
            }

    ds = datasets.Dataset.from_generator(gen)
    assert isinstance(ds, datasets.Dataset)

    ds = ds.filter(dsutil.make_1d_length_filter_fn("values", max_len=6))
    assert all(length <= 6 for length in ds["length"])
    assert len(ds) == 8
