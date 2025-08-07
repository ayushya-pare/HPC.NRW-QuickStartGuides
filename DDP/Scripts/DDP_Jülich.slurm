#!/bin/bash

#SBATCH --partition=sgpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --job-name=ddp-jülich-mnist
#SBATCH --output=logs/ddp_julich_%j.out
#SBATCH --error=logs/ddp_julich_%j.err


# Set up master address/port for distributed
export MASTER_ADDR='localhost'
export MASTER_PORT=12355


# Run with torchrun
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py "$@"
