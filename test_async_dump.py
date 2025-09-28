import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

CHECKPOINT_DIR = "checkpoint"


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # create a model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model = fully_shard(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    future_list = []
    for step in range(3):
        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        
        os.makedirs(f"checkpoint/model_step_{step}", exist_ok=True)
        os.makedirs(f"checkpoint/opt_step_{step}", exist_ok=True)
        writer_1 = dcp.FileSystemWriter(
            path=f"checkpoint/model_step_{step}",
            overwrite=True,
            single_file_per_rank=True,
        )
        write_2 = dcp.FileSystemWriter(
            path=f"checkpoint/opt_step_{step}",
            overwrite=True,
            single_file_per_rank=True,
        )

        checkpoint_future_1 = dcp.async_save(model.state_dict(), storage_writer=writer_1, async_checkpointer_type=AsyncCheckpointerType.PROCESS)
        checkpoint_future_2 = dcp.async_save(optimizer.state_dict(), storage_writer=write_2, async_checkpointer_type=AsyncCheckpointerType.PROCESS)
        future_list.append(checkpoint_future_1)
        future_list.append(checkpoint_future_2)

    for item in future_list:
        item.result()
    import time
    time.sleep(4)
    dist.destroy_process_group(pg)
    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    world_size = 2
    print(f"Running async checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )