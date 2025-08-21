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
