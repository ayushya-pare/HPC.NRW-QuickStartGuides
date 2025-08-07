# Imports
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# Setup Port and address environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def gpu_info(rank):
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    print(f"[DDP] Rank {rank} using GPU {idx}: {name}")

    

#CNN for FashionMNIST
class CNN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Training loop
def train_mnist(rank, world_size, epochs=5):
    setup(rank, world_size)
    gpu_info(rank)
    torch.manual_seed(0)
    batch_size = 64

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST('.', download=True, train=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    model = CNN_MNIST().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(rank), target.to(rank)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss {loss.item()}")
        if rank == 0:
            print(f"Epoch {epoch}, Avg Loss: {epoch_loss/len(dataloader)}")
    cleanup()
    

    
# Main function call
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count() 
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(train_mnist, args=(world_size,), nprocs=world_size, join=True)
    
