import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def gpu_info(rank, dev0, dev1):
    print(f"[DDP] Rank {rank} using GPUs {dev0} and {dev1}:",
          torch.cuda.get_device_name(dev0), "&", torch.cuda.get_device_name(dev1))

class GPT2MpModel(torch.nn.Module):
    def __init__(self, dev0, dev1):
        super().__init__()
        # Load GPT2
        self.device0 = dev0
        self.device1 = dev1
        config = GPT2Config.from_pretrained("gpt2")
        self.transformer = GPT2LMHeadModel(config).transformer
        self.lm_head = GPT2LMHeadModel(config).lm_head

        # Move blocks 0-5 to dev0, 6-11 to dev1 for GPT2 Small
        for i, block in enumerate(self.transformer.h):
            if i < len(self.transformer.h)//2:
                block.to(dev0)
            else:
                block.to(dev1)
        self.transformer.wte.to(dev0)
        self.transformer.wpe.to(dev0)
        self.transformer.ln_f.to(dev1)
        self.lm_head.to(dev1)

    def forward(self, input_ids):
        # Embedding on dev0
        x = input_ids.to(self.device0)
        inputs_embeds = self.transformer.wte(x) + self.transformer.wpe(torch.arange(x.size(1), device=self.device0))
        hidden_states = inputs_embeds

        # First half of blocks on dev0
        for i, block in enumerate(self.transformer.h):
            if i < len(self.transformer.h)//2:
                hidden_states = block(hidden_states)[0]
        # Move hidden_states to dev1
        hidden_states = hidden_states.to(self.device1)

        # Second half of blocks on dev1
        for i, block in enumerate(self.transformer.h):
            if i >= len(self.transformer.h)//2:
                hidden_states = block(hidden_states)[0]

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

def demo_gpt2_model_parallel(rank, world_size):
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    print(f"Running GPT-2 Model Parallel DDP Example on rank {rank}.")
    setup(rank, world_size)
    gpu_info(rank, dev0, dev1)

    model = GPT2MpModel(dev0, dev1)
    ddp_model = DDP(model)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt = "Deep learning is"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]

    # Forward pass (logits only, not generation)
    logits = ddp_model(input_ids)
    print(f"[Rank {rank}] Logits shape: {logits.shape}")

    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    print("Available GPUs: ", n_gpus)
    assert n_gpus >= 2 and n_gpus % 2 == 0, f"Requires at least 2 GPUs and even number to run, but got {n_gpus}"
    world_size = n_gpus // 2
    mp.spawn(demo_gpt2_model_parallel, args=(world_size,), nprocs=world_size, join=True)
