import atexit
import functools
import os

import torch
import torch.distributed as dist


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


@functools.cache
def _is_ddp_available():
    """
    Checks if distributed data parallel (DDP) is available.

    Returns:
        bool: True if DDP is available, False otherwise.
    """
    required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    return all(k in os.environ for k in required_vars)


def initialize_ddp() -> tuple[int, int]:
    """
    Initializes the distributed data parallel (DDP) environment.

    Returns:
        tuple[int, int]: The local rank and world size.
    """
    if not _is_ddp_available():
        return 0, 1

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()

    if torch.cuda.is_available():
        if torch.cuda.device_count() != world_size:
            raise ValueError(
                "The program must be run with N processes where N is the number of cuda devices."
            )

        torch.cuda.set_device(local_rank)

    atexit.register(lambda: dist.destroy_process_group())

    return local_rank, world_size
