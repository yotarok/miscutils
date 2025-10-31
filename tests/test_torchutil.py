import torch

from miscutils import torchutil


def test_module_summary():
    mod = torch.nn.Linear(10, 10)
    summary = torchutil.ModuleSummary.from_module(mod)
    expected_str = """
MODULE: 110 <Linear>
  weight: 10x10 = 100
  bias: 10 = 10
"""
    expected_count = 110
    assert summary.total_parameter_count == expected_count
    assert str(summary).strip() == expected_str.strip()


def test_module_summary_2():
    mod = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.LayerNorm(5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 7),
    )
    summary = torchutil.ModuleSummary.from_module(mod)
    expected_str = """
MODULE: 72 <Sequential>
  0: 20 <Linear>
    weight: 5x3 = 15
    bias: 5 = 5
  1: 10 <LayerNorm>
    weight: 5 = 5
    bias: 5 = 5
  2: 0 <ReLU>
  3: 42 <Linear>
    weight: 7x5 = 35
    bias: 7 = 7
"""
    expected_count = 72
    assert summary.total_parameter_count == expected_count
    assert str(summary).strip() == expected_str.strip()


def test_module_summary_json_dump():
    mod = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 7),
    )
    json_dict = torchutil.ModuleSummary.from_module(mod).to_json_dict()
    assert json_dict == {
        "0: Linear": {
            "weight: Parameter": {
                "shape": [5, 3],
                "dtype": "torch.float32",
            },
            "bias: Parameter": {
                "shape": [5],
                "dtype": "torch.float32",
            },
        },
        "1: ReLU": {},
        "2: Linear": {
            "weight: Parameter": {
                "shape": [7, 5],
                "dtype": "torch.float32",
            },
            "bias: Parameter": {
                "shape": [7],
                "dtype": "torch.float32",
            },
        },
    }
