from contextlib import contextmanager

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
