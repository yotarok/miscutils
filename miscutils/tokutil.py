import typing

import torch


class Detokenizer(typing.Protocol):
    def decode(self, token_ids: list[int], **kwargs) -> str: ...


def detokenize_batch(
    tok: Detokenizer, ids: torch.Tensor, mask: torch.Tensor, **decode_kwargs
) -> list[str]:
    if ids.ndim != 2:
        raise ValueError("shape of `ids` must be `(batch_size, max_seq_len)`")
    if ids.shape != mask.shape:
        raise ValueError(
            f"shapes of `ids` and `mask` must match ({ids.shape} != {mask.shape})"
        )
    id_lists = [
        [i for i, m in zip(row_ids, row_mask) if m]
        for row_ids, row_mask in zip(ids, mask)
    ]
    return [tok.decode(row, **decode_kwargs) for row in id_lists]
