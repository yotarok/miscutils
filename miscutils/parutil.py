import torch


def all_reduce(x: torch.Tensor, opname: str) -> torch.Tensor:
    """
    Performs an all-reduce operation with the given reduce operation.

    This function provides a compatibility layer between accelerate and
    torch.distributed.

    Args:
        x (torch.Tensor): The tensor to reduce.
        opname (str): The name of the reduce operation (e.g., 'sum', 'mean').

    Returns:
        torch.Tensor: The reduced tensor.
    """
    if not _is_ddp_available():
        return x

    ret_device = x.device
    if torch.cuda.is_available():
        # all reduce doesn't work if it's not located in GPU
        x = x.to(device="cuda")
    tag, prefn, postfn = _OPNAME_TO_IMPL[opname.lower()]
    x = x.detach()
    if prefn is not None:
        x = prefn(x)

    dist.all_reduce(x, op=tag)

    if postfn is not None:
        x = postfn(x)
    return x.to(ret_device)
