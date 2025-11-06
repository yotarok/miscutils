import typing

import torch


class Detokenizer(typing.Protocol):
    def decode(self, token_ids: list[int], **kwargs) -> str: ...
    def batch_decode(self, token_ids: list[int] | list[list[int]], **kwargs) -> str: ...


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
        [int(i.item()) for i, m in zip(row_ids, row_mask) if m]
        for row_ids, row_mask in zip(ids, mask)
    ]
    ret = tok.batch_decode(id_lists, **decode_kwargs)
    assert isinstance(ret, list)
    return ret
