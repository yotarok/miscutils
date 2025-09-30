import typing

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
