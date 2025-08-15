
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

# ---------- Device header ----------
            
def get_device(rank, world_size):
    if rank == 0:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"[Device] Detected {n} CUDA device(s):")
            for i in range(n):
                print(f"  - cuda:{i}: {torch.cuda.get_device_name(i)}")
            torch.cuda.set_device(rank)

        else:
            print("[Device] CUDA not available — using CPU")

# ---------- DDP setup / teardown ----------
# Setup Port and address environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

# ---------- Model ----------
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

# ---------- Training (per-rank) ----------
def train_mnist(rank, world_size, epochs=5, batch_size=64):
    device = get_device(rank, world_size)
    
    setup(rank, world_size)
    torch.manual_seed(0)
#    torch.backends.cudnn.benchmark = True

    tfm = transforms.ToTensor()
    root = "Data"  # parent of FashionMNIST/

    train_ds = datasets.FashionMNIST('./Data', train=True, download=False, transform=tfm)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device(f"cuda:{rank}")

    model = CNN_MNIST().to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        ddp_model.train()
        train_sampler.set_epoch(epoch)

        dist.barrier(device_ids=[rank])

        start = time.time()
        
        total_local = 0
        loss_sum_local = 0.0
        correct_local = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            n = xb.size(0)
            total_local    += n
            loss_sum_local += loss.item() * n

            correct_local  += (logits.argmax(1) == yb).sum().item()

        # epoch time = max across ranks
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        dist.barrier(device_ids=[rank])
        epoch_time = time.time() - start  # ~max across ranks due to barrier

        # rank-0 prints *its own shard* metrics (approximate global)
        if rank == 0:
            avg_loss = loss_sum_local / total_local
            acc = correct_local / total_local
            print(f"Epoch {epoch:02d}/{epochs} | loss: {avg_loss:.4f} | acc: {acc:.4f} | time: {epoch_time:.2f}s")

    cleanup()

# ---------- Main (your requested form) ----------
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"

    mp.spawn(train_mnist, args=(world_size,), nprocs=world_size, join=True)
