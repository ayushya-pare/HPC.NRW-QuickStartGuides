#!/bin/bash
#SBATCH --job-name=ddp-basic
#SBATCH --output=logs/ddp_%j.out
#SBATCH --error=logs/ddp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=sgpu_devel
#SBATCH --time=00:20:00

## Activate environment
conda deactivate
source venv_ddp/bin/activate

## Run the training
python Scripts/Basic_DDP.py
##python Scripts/MNIST.py
##python Scripts/MiniGPT.py
