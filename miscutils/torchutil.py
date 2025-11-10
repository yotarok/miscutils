from contextlib import contextmanager
import dataclasses
from typing import Any

import torch
from torch import nn


def freeze_module(mod: nn.Module) -> None:
    """
    Freezes the parameters of a given module.

    Args:
        mod (nn.Module): The module whose parameters are to be frozen.
    """
    for p in mod.parameters():
        p.requires_grad = False


@contextmanager
def eval_context(*modules):
    """Temporarily switches to eval mode.

    ```
    with eval_context(m):
        ...
    ```
    is equivalent with
    ```
    state = m.training
    m.eval()
    ...
    m.train(state)
    ```

    Therefore, for some cases where the above simple switching does not work,
    e.g. when `m.eval()` has a side-effect, or a sub-module of m has an
    independent state, this context manager does not work.
    """
    states = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield modules
    finally:
        for m, state in zip(modules, states):
            m.train(state)


@dataclasses.dataclass
class ModuleSummary:
    name: str
    cls_name: str
    total_parameter_count: int
    parameters: dict[str, torch.Tensor]  # meta-tensors
    children: dict[str, "ModuleSummary"]
    # TODO: buffers are not supported currently

    def to_string(self, *, prefix: str = "") -> str:
        ret = ""
        for parname, p in self.parameters.items():
            shape_string = "x".join(str(i) for i in p.shape)
            ret += f"{prefix}  {parname}: {shape_string} = {p.numel()}\n"

        for submod in self.children.values():
            ret += submod.to_string(prefix=prefix + "  ")
        ret = (
            f"{prefix}{self.name}: {self.total_parameter_count} <{self.cls_name}>\n"
            + ret
        )
        return ret

    def __str__(self) -> str:
        return self.to_string()

    def to_json_dict(self, with_total_parameter_count: bool = True) -> dict[str, Any]:
        """Converts self into a compact JSON format (suitable for experiment trackers)."""
        ret: dict[str, Any] = (
            {"total_parameter_count": self.total_parameter_count}
            if with_total_parameter_count
            else {}
        )

        for name, p in self.parameters.items():
            ret[name + ": Parameter"] = {
                "shape": list(p.shape),
                "dtype": str(p.dtype),
            }
        for name, m in self.children.items():
            ret[name + ": " + m.cls_name] = m.to_json_dict()
        return ret

    @classmethod
    def from_module(cls, module: nn.Module, name: str = "MODULE") -> "ModuleSummary":
        cls_name = type(module).__qualname__
        total_parameter_count = 0
        parameters = {}
        children = {}
        for parname, p in module.named_parameters(recurse=False):
            total_parameter_count += p.numel()
            parameters[parname] = p.to("meta")

        for subname, submod in module.named_children():
            children[subname] = cls.from_module(submod, subname)
            total_parameter_count += children[subname].total_parameter_count

        return cls(
            name=name,
            cls_name=cls_name,
            total_parameter_count=total_parameter_count,
            parameters=parameters,
            children=children,
        )
