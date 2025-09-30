from torch import nn


def freeze_module(mod: nn.Module) -> None:
    """
    Freezes the parameters of a given module.

    Args:
        mod (nn.Module): The module whose parameters are to be frozen.
    """
    for p in mod.parameters():
        p.requires_grad = False
