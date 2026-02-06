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


### Next-Up
Scaling DDP to larger models like MiniGPT, combining DDP with model parallelism.

--- 
Contributions: 
- Ayushya Pare (Research assistant, University of Bonn)
- Junaid Mir (Research assistant, University Duisburg-Essen)
