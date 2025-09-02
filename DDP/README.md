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
DDP/
│── Scripts/
│   ├── MNIST.py          # DDP-enabled version
│   └── cnn_model.py      # CNN_MNIST class
│   └── Basic_DDP.slurm   # SLURM script for multi-GPU DDP
│── requirements.txt
│── Data/                 # FashionMNIST dataset (pre-downloaded)
│── logs/                 # Training logs
```

---
## Reproducing

This repository contains the source code used to train a CNN model on FashionMNIST using PyTorch’s **Distributed Data Parallel (DDP)**. The entire implementation is written in PyTorch, and we make use of auxiliary libraries such as **Torchvision** for dataset handling and transforms, and **SLURM** job scripts for execution on HPC clusters.

All package requirements are listed in the `requirements.txt` file. The Python version used is **3.10+**.

Since our experiments were performed on the **Marvin HPC cluster** at the University of Bonn, which uses **SLURM** for job scheduling, all major training runs were launched via the provided **bash scripts**, which can be found in the `/Scripts` folder.

## Steps to Reproduce

1. Clone this repository and set up a Python virtual environment.
2. Install dependencies from `requirements.txt`.
3. Submit the provided SLURM script (e.g., `Basic_DDP.slurm`) to your cluster.
4. Logs for training output and GPU utilization will be saved automatically in the `/logs` directory.

### Clone repository
```bash
git clone git@github.com:ayushya-pare/HPC.NRW-QuickStartGuides.git
cd HPC.NRW-QuickStartGuides/DDP
```

### Install dependencies
Install the dependencies from the ```requirements.txt``` file. Run the following commands inside your HPC virtual environment:

```bash
module load CUDA/12.6.0 
python3 -m venv venv_ddp
source venv_ddp/bin/activate
pip install -r requirements.txt
```
**Note**: # Module names/versions vary by cluster — check module avail or your cluster docs

### Run script 
1. Prepare SLURM commands. Open ```Scripts/Basic_DDP.slurm``` and adjust the parameters.
    ```bash
    #!/bin/bash
    #SBATCH --job-name=ddp-fmnist
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=2
    #SBATCH --cpus-per-task=8
    #SBATCH --gres=gpu:2
    #SBATCH --partition=sgpu_devel ## depends on your cluster
    #SBATCH --time=00:20:00
    #SBATCH --output=logs/%x_%j.out
    
    . 
    .
    .
    ```
3. Submit the SLURM script
    ```bash
    sbatch Scripts/Basic_DDP.slurm
    ```

### Monitor logs

- Training logs and metrics are saved in the ```logs/``` directory as ```logs/<jobname>_<jobid>.out```.
    Inside you’ll see something like this:
    ```yaml
    [Device] Detected 2 CUDA device(s):
    cuda:0: NVIDIA A100-SXM4-80GB | [host=sgpu014]
    cuda:1: NVIDIA A100-SXM4-80GB | [host=sgpu014]
    ```

    ```yaml
    [rank=1/1] [gpu=1] | Epoch 01/15 | loss: 0.5573 | acc: 0.8029
    [rank=0/1] [gpu=0] | Epoch 01/15 | loss: 0.5636 | acc: 0.7967
    .
    .
    ```
    The script sees 2 NVIDIA A100 GPUs (cuda:0 and cuda:1) on node sgpu014.
    It assigns rank 0 → GPU 0 and rank 1 → GPU 1.The two processes (ranks) are working in parallel, with similar losses and accuracies, which confirms DDP is running correctly. Small differences in loss/accuracy per rank are expected because each GPU sees different data shards each epoch.

- GPU utilization is recorded in ```logs/gpu_<jobid>.log```, which looks like this:
    ```yaml
    # gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk     fb   bar1
    # Idx      W      C      C      %      %      %      %      %      %    MHz    MHz     MB     MB
        0     64     30     46      0      0      0      0      0      0   1593    210      4      1
        1     61     28     44      0      0      0      0      0      0   1593    210      4      1
        0     76     30     46      0      0      0      0      0      0   1593   1275    452      4
        1     61     28     44      0      0      0      0      0      0   1593   1275     24      2
        0     87     32     45      0      0      0      0      0      0   1593   1410    916      4
    .
    .
    ```

    GPU logs show how each GPU’s resources are being used over time — power, temperature, memory, and compute (SM) utilization.
    They help verify whether GPUs are idle or fully engaged during training, and whether memory and power usage match expectations         for the workload.


## Implementation with DDP
The following sections show how the ```MNIST.py``` training script and the ```Basic_DDP.slurm``` job script were designed using PyTorch Distributed Data Parallel (DDP). The Python code sets up the model, communication, and training loop across multiple GPUs, while the SLURM script ensures correct execution on the HPC cluster with resource allocation and logging.

### Step 1. Packages

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

### Step 2. Setup Communication

Every DDP run needs:

* `MASTER_ADDR`: address of the master node (localhost if single node).
* `MASTER_PORT`: free port for process group.
* `rank`: unique ID for this process (0 … world\_size-1).
* `world_size`: total number of processes (GPUs across all nodes).

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

### Step 3. Define CNN Model

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

### Step 4. Training Loop (Per Process/Rank)

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
The dataset is already downloaded at ```Data/FashionMNIST/*```, keep ```download=False```.
If downloading for the first time, set ```download=True``` once to fetch it automatically.

### Step 5. Main Block

```python
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    assert world_size >= 2, f"Need at least 2 GPUs, found {world_size}"
    mp.spawn(train_mnist, args=(world_size,), nprocs=world_size, join=True)
```

---
Note: World size = number of GPUs you requested per node × nodes
  

### Step 6. SLURM Scripts

```bash
#!/bin/bash
#SBATCH --job-name=ddp-fmnist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=sgpu_devel ## depends on your cluster
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out

# GPU utilization logging
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi dmon -h >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -d 5 > logs/gpu_${SLURM_JOB_ID}_dmon.log 2>&1 &
  else
    nvidia-smi -l 5 > logs/gpu_${SLURM_JOB_ID}.log 2>&1 &
  fi
  GPULOG_PID=$!
  trap "kill ${GPULOG_PID} 2>/dev/null || true" EXIT
fi

# Single-node defaults for DDP
export MASTER_ADDR=localhost
export MASTER_PORT=$(shuf -i 20000-65000 -n1)

python Scripts/MNIST.py

```
---
Notes: Mapping rank to GPU

Using ``` ddp_model = DDP(model, device_ids=[rank], output_device=rank)``` assumes global ```rank == local GPU id```. This is true for single-node when you spawn one process per GPU and ranks are 0..N-1.

**Multi-node caution**: global rank 0 is not necessarily ```cuda:0``` on every node. Use ```local_rank``` (e.g., ```os.environ["LOCAL_RANK"]```) to select the device:

```python
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")
ddp_model = DDP(model.to(device), device_ids=[local_rank], output_device=local_rank)
```
This is handled automatically when you launch with ```torchrun```.

---

### Next-Up
Scaling DDP to larger models like MiniGPT, combining DDP with model parallelism.

--- 
Contributions: 
- Ayushya Pare (Research assistant, University of Bonn)
- Junaid Mir (Research assistant, University Duisburg-Essen)
