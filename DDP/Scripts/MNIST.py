# MNIST_DDP.py  — uses mp.spawn(..., args=(world_size,), ...) as requested
import os, time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

# ---------- Device header ----------
def print_device_header(rank, world_size):
    if rank == 0:
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"[Device] Detected {n} CUDA device(s):")
            for i in range(n):
                print(f"  - cuda:{i}: {torch.cuda.get_device_name(i)}")
            used = ", ".join([f"cuda:{i}" for i in range(min(world_size, n))])
            print(f"[Device] Using {used}")
        else:
            print("[Device] CUDA not available — using CPU")

# ---------- DDP setup / teardown ----------
# Setup Port and address environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
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
    setup(rank, world_size)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    if rank == 0:
        print_device_header(rank, world_size)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    # Download only on rank 0 to avoid races
    if rank == 0:
        datasets.FashionMNIST('.', download=True, train=True, transform=transform)
    try:
        dist.barrier(device_ids=[rank])  # quiets NCCL warnings on newer PyTorch
    except TypeError:
        dist.barrier()

    dataset = datasets.FashionMNIST('.', download=False, train=True, transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    model = CNN_MNIST().to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        ddp_model.train()
        sampler.set_epoch(epoch)

        start = time.time()
        total_local = 0
        loss_sum_local = 0.0
        correct_local = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bsz = xb.size(0)
            total_local    += bsz
            loss_sum_local += loss.item() * bsz
            correct_local  += (logits.argmax(1) == yb).sum().item()

        # epoch time = max across ranks
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        epoch_time_local = torch.tensor([time.time() - start], dtype=torch.float32, device=device)

        # Reduce metrics (SUM) and time (MAX)
        t_total = torch.tensor([total_local], dtype=torch.long, device=device)
        t_loss  = torch.tensor([loss_sum_local], dtype=torch.float32, device=device)
        t_corr  = torch.tensor([correct_local], dtype=torch.long, device=device)

        dist.all_reduce(t_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_loss,  op=dist.ReduceOp.SUM)
        dist.all_reduce(t_corr,  op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_time_local, op=dist.ReduceOp.MAX)

        if rank == 0:
            total_all = t_total.item()
            avg_loss  = t_loss.item() / total_all
            acc       = t_corr.item() / total_all
            t_epoch   = epoch_time_local.item()
            print(f"Epoch {epoch:02d}/{epochs} | loss: {avg_loss:.4f} | acc: {acc:.4f} | time: {t_epoch:.2f}s")

    cleanup()

# ---------- Main (your requested form) ----------
if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    mp.spawn(train_mnist, args=(world_size,), nprocs=world_size, join=True)
