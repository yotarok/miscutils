from collections.abc import Sequence
import dataclasses
import typing
from typing import Any, Literal, TypeAlias, NamedTuple

import numpy as np
import numpy.typing as npt
import torch

N = typing.TypeVar("N", bound=np.generic)


def right_pad_to_multiple_of(
    n: int, x: npt.NDArray, fill_value: N, axis: int = 0
) -> npt.NDArray[N]:
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
    xs: list[torch.Tensor], *, value=0.0, max_length: int | None = None
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
    if max_length is None:
        max_length = max(lengths)

    # TODO: some treatments for oversized samples (error/ truncate/ extend_result)
    #       note that we shouldn't support filtering here. it's outside of responsibility.
    padded_x = [
        torch.nn.functional.pad(
            x,
            [0, max_length - x.shape[dim]],
            value=value,
        )
        for x in xs
    ]
    mask = length_to_right_pad_mask(
        typing.cast(torch.LongTensor, torch.tensor(lengths, device=xs[0].device)),
        max_length,
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


PadDirection: TypeAlias = Literal["left"] | Literal["right"]


class _CollatePadSpec(NamedTuple):
    direction: PadDirection
    pad_value: Any
    mask_name: str | None
    max_length: int | None


@dataclasses.dataclass(frozen=True)
class PadAndCollateFn:
    pad_value: Any = None
    pad_direction: PadDirection = "right"
    pad_columns: dict[str, _CollatePadSpec] = dataclasses.field(default_factory=dict)
    output_all_columns: bool = False

    def include_all_columns(self, include: bool = True) -> "PadAndCollateFn":
        return dataclasses.replace(self, output_all_columns=include)

    def pad_column(
        self,
        column_name: str,
        mask_name: str | None,
        pad_direction: PadDirection | None = None,
        pad_value: Any = None,
        max_length: int | None = None,
    ) -> "PadAndCollateFn":
        pad_columns = self.pad_columns
        pad_direction = (
            pad_direction if pad_direction is not None else self.pad_direction
        )
        pad_value = pad_value if pad_value is not None else self.pad_value
        if pad_value is None:
            raise ValueError("No default `pad_value` is set for this collator.")
        pad_columns[column_name] = _CollatePadSpec(
            pad_direction, pad_value, mask_name, max_length
        )
        return dataclasses.replace(self, pad_columns=pad_columns)

    def __call__(self, rows):
        ret = dict()
        for col, spec in self.pad_columns.items():
            if spec.direction != "right":
                raise ValueError("Only right padding is supported currently.")
            pad_fn = right_pad if spec.direction == "right" else right_pad

            values, mask = pad_fn(
                [row[col] for row in rows],
                max_length=spec.max_length,
            )
            ret[col] = values
            if spec.mask_name is not None:
                ret[spec.mask_name] = mask
        if self.output_all_columns and len(rows) > 0:
            ks = rows[0].keys()
            for k in ks:
                if k in self.pad_columns.keys():
                    continue
                ret[k] = [row[k] for row in rows]
        return ret
