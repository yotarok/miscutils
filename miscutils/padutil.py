import typing
from typing import Sequence

import numpy as np
import numpy.typing as npt
import torch

T = typing.TypeVar("T")


def right_pad_to_multiple_of(
    n: int, x: npt.NDArray, fill_value: T, axis: int = 0
) -> npt.NDArray[T]:
    length = x.shape[axis]
    target_length = int(np.ceil(length / n)) * n
    npad = target_length - length
    assert npad >= 0

    if npad == 0:
        return x

    paddings_shape = (*x.shape[:axis], npad, *x.shape[axis + 1 :])
    paddings = np.full(paddings_shape, fill_value, dtype=x.dtype)
    return np.concatenate((x, paddings), axis=axis)


def length_to_right_pad_mask(
    lengths: torch.LongTensor, max_length: int
) -> torch.BoolTensor:
    """
    Converts lengths to a right-padded mask.

    Args:
        lengths (torch.LongTensor): The lengths to convert.
        max_length (int): The maximum length for padding.

    Returns:
        torch.BoolTensor: The right-padded mask.
    """
    idxs = torch.arange(max_length, device=lengths.device).reshape(
        (1,) * (lengths.ndim - 1) + (-1,)
    )
    return typing.cast(torch.BoolTensor, idxs < lengths[..., None])


def right_pad_mask_to_length(
    mask: torch.Tensor, verify: bool = True
) -> torch.LongTensor:
    """
    Converts a right-padded mask to lengths.

    Args:
        mask (torch.Tensor): The mask to convert.
        verify (bool): Whether to verify the input mask.

    Returns:
        torch.LongTensor: The lengths.
    """
    ret = typing.cast(torch.LongTensor, mask.long().sum(dim=-1))
    # this function also verifies the input
    if verify:
        reconstructed_mask = length_to_right_pad_mask(ret, max_length=mask.shape[-1])
        if not (mask == reconstructed_mask).all().item():
            raise ValueError(
                "This function is expected to be called with right-padded masks"
            )
    return ret


L = typing.TypeVar("L", bound=int | torch.Tensor | npt.NDArray)


def simulate_conv1d_length(
    length: L, kernel_size: int | Sequence[int], stride: int | Sequence[int]
) -> L:
    if isinstance(kernel_size, Sequence) or isinstance(stride, Sequence):
        if not (isinstance(kernel_size, Sequence) and isinstance(stride, Sequence)):
            raise ValueError(
                "`kernel_size` and `stride` must be both sequece or scalar."
            )
        if len(kernel_size) != len(stride):
            raise ValueError("The lengths of `kernel_size` and `stride` must match.")

        for ks, st in zip(kernel_size, stride):
            length = simulate_conv1d_length(length, ks, st)
        return length

    return typing.cast(L, ((length - kernel_size) // stride) + 1)


def right_pad(
    xs: list[torch.Tensor], *, value=0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Makes a padded batch from a sequence of tensors.

    Returns:
      `(y, mask)` where `y` is the array padded to fit the longest array in `xs`, `mask`
      is the bool array indicating which elements in in `y` are non-padded elements.
    """
    if not xs:
        raise ValueError("Attempted to make a padded batch from an empty sequence")
    dim = -1  # TODO: Generalize to arbitrary dim.
    lengths = [x.shape[dim] for x in xs]
    max_length = max(lengths)
    padded_x = [
        torch.nn.functional.pad(
            x,
            [0, max_length - x.shape[dim]],
            value=value,
        )
        for x in xs
    ]
    mask = length_to_right_pad_mask(
        torch.tensor(lengths, device=xs[0].device), max_length
    )
    return torch.stack(padded_x), mask


def mask_last_n_elems(n: int, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Masks the last `n` unmasked elements from the mask array."""
    mask = mask.bool().long()
    return torch.logical_and(mask, mask.flip(dim).cumsum(dim).flip(dim) > n)


def pack_and_pad_right(
    x: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim != -1 or x.ndim != 2 or mask.ndim != 2:
        raise ValueError("Currently, only [batch_size, seq_len] shape is supported.")

    t_idxs = torch.where(mask, mask.bool().long().cumsum(dim) - 1, -1)
    # ^ Note that disposing unused elements to "-1" is not problematic because
    # if there's an unsed element, "-1" is vacant, so can be safely used as a trash bin.

    instance_idxs = torch.arange(x.shape[0])[:, None]

    y = torch.empty_like(x)
    y[instance_idxs, t_idxs] = x

    lengths = mask.long().sum(-1, keepdim=True)
    ymask = (
        torch.arange(y.shape[dim], dtype=torch.int32, device=y.device)[None, :]
        < lengths
    )
    return y, ymask
