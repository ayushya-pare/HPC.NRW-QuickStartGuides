# HPC-NRW-AP4-QuickStartGuides

## Distributed Data Parallel (DDP) with PyTorch on HPC

Distributed Data Parallel (DDP) is a powerful module in PyTorch that allows you to parallelize your model across multiple GPUs or even multiple machines, making it perfect for large-scale deep learning applications.

In this guide, we implement DDP to train a CNN model on FashionMNIST. This guide entails:
- What DDP does and how it works.
- Repository structure and setup for HPC clusters.
- Multi-GPU training with DDP on Uni-Bonn HPC cluster "Marvin".
- Monitoring metrics (loss, accuracy) and GPU utilization.

The torch.distributed package provides collective communication primitives that make multi-GPU training possible. In DDP, each GPU (or process) runs a full copy of the model, computes local gradients, and synchronizes them with the other processes. This synchronization happens during the backward pass through autograd hooks, which ensures that every process ends up with the same gradients. The result is consistent gradients across all processes, leading to a correct global model update.

The repository structure looks like this:

```
HPC-DDP-Tutorial/
│── Scripts/
│   ├── MNIST.py          # DDP-enabled version
│   └── model.py          # CNN_MNIST class
│
│── run_ddp.slurm         # SLURM script for multi-GPU DDP
│── requirements.txt
│── Data/                 # FashionMNIST dataset (pre-downloaded)
│── logs/                 # Training logs
```

---

## Requirements and Installation

**requirements.txt**

```txt
torch>=2.0
torchvision>=0.15
```

Install inside your HPC virtual environment:

```bash
module load cuda/12.0  # (depends on your cluster)
python3 -m venv venv_ddp
source venv_ddp/bin/activate
pip install -r requirements.txt
```

---

## Step 1. Packages

```python
import os, time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
```

---

## Step 2. Setup Communication

Every DDP run needs:

* `MASTER_ADDR`: address of the master node (localhost if single node).
* `MASTER_PORT`: free port for process group.
* `rank`: unique ID for this process (0 … world\_size-1).
* `world_size`: total number of processes (GPUs).

```python
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

> Use `"nccl"` backend for GPU clusters.

---

## Step 3. Define CNN Model

```python
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
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
```

---

## Step 4. Training Loop (Per Process/Rank)

Each process:

* loads a **sharded dataset** via `DistributedSampler`.
* trains its own replica of the model.
* gradients are **synchronized** during `.backward()`.

```python
def train_mnist(rank, world_size, epochs=5, batch_size=64):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(0)

    tfm = transforms.ToTensor()
    train_ds = datasets.FashionMNIST('./Data', train=True, download=False, transform=tfm)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    model = CNN_MNIST().to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        dist.barrier()  # sync start
        start = time.time()

        total, loss_sum, correct = 0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            n = yb.size(0)
            total    += n
            loss_sum += loss.item() * n
            correct  += (logits.argmax(1) == yb).sum().item()

        dist.barrier()  # sync end
        epoch_time = time.time() - start

        if rank == 0:  # log only once
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"loss: {loss_sum/total:.4f} | acc: {correct/total:.4f} | "
                  f"time: {epoch_time:.2f}s")

    cleanup()
```

---

## Step 5. Main Block

```python
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Need at least 2 GPUs, found {world_size}"
    mp.spawn(train_mnist, args=(world_size,), nprocs=world_size, join=True)
```

---

## Step 6. SLURM Scripts

```bash
#!/bin/bash
#SBATCH --job-name=ddp-fmnist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=develbooster # (depends on your cluster)
#SBATCH --time=00:20:00
#SBATCH --output=logs_ddp_%j.out

source venv_ddp/bin/activate
python Scripts/MNIST.py
```

---

## Notes
* **torchrun**: modern alternative to `mp.spawn`. Example:

```bash
torchrun --nproc_per_node=4 Scripts/MNIST.py
```

---

## Next-Up
Scaling DDP to larger models like MiniGPT, combining DDP with model parallelism.
