import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# Used in training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def spawn_fn(fn, world_size, *artifacts):
    mp.spawn(fn, args=(world_size, *artifacts), nprocs=world_size, join=True)


def unwrap(model):
    return model.module if isinstance(model, DDP) else model
