import atexit
import functools
import os

import torch
import torch.distributed as dist


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
