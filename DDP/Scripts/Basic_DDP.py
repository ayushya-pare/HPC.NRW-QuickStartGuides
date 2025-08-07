# imports
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# setup DDP environment (Adress and Port)
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


# Create a Toy Model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10) # Layer 1 :
        self.relu = nn.ReLU() # Activation
        self.net2 = nn.Linear(10, 5) # Output

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))



# Train the model (on multiple GPUs)

def demo_basic(rank, world_size):
    # setup environment variables (Address port and process group)
#    print(f"Running basic DDP example on rank {rank}.")
#    setup(rank, world_size)

    # alternate way to setup environment ()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Running basic DDP example on rank {rank}.")


    # instantiate the model - and allocate a GPU
    model = ToyModel().to(rank) # A copy of the model will be allocated to one GPU
    ddp_model = DDP(model, device_ids=[rank]) # instantiate the model for distributed data on GPU

    # define loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Execute the training
    optimizer.zero_grad()
    inputs = torch.randn(20,10)
    outputs = ddp_model(inputs)

    labels = torch.randn(20, 5).to(rank) # labels allocated to the corresponding GPUs
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    print("Loss : ", loss)

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


# main (Run DDP using multiprocessing)
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(demo_basic,args=(world_size,),nprocs=world_size,join=True) # On each of the available GPUs, the process will be spawned

