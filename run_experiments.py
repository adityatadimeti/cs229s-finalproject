import subprocess
import ast
import json
import time

import os
from torch.nn import functional as F
import time
import tiktoken
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

import torch.nn.utils.prune as prune # pruning library
from tqdm import tqdm # progress bar library
#tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
#memory_usage = torch.cuda.max_memory_allocated()


out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'gpt2' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'cs229s'
wandb_run_name = 'gpt2' # 'run' + str(time.time())      #change this to gpt2-medium when running speculative
# data

gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 10 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 1 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
num_samples = 1 # number of samples to draw
dataset = 'shakespeare'
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8


def quantize_param(param): #this data is the param data

    rangemin = -128
    rangemax = 127
    
    maxval = torch.max(torch.abs(param))
    scaling_factor = rangemax / maxval
    scaled_tensor = param * scaling_factor

    clipped_tensor = torch.clamp(scaled_tensor, rangemin, rangemax)
    quantized_tensor = clipped_tensor.to(torch.int8)
    quantized_tensor.requires_grad = False
    return quantized_tensor, maxval
    

def dequantize_param(param, maxval):
    rangemax = 127
    dequantized_tensor = param / rangemax * maxval
    dequantized_tensor = dequantized_tensor.to(torch.float32)
    return dequantized_tensor


ckpt_path = os.path.join("pretrained", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

print("quantizing the model parameters")
max_vals = {} #this is a dictionary that maps the name of the parameter to the max value of that parameter
print("param names")
for name, param in model.named_parameters():
    if ".wte." in name or ".wpe." in name or "weight" in name or "bias" in name:
        quantized_tensor, maxval = quantize_param(param.data)
        #print(name + "_maxval")
        max_vals[name+"_maxval"] = maxval
        #assign the quantized tensor back to the model parameter
        #print(f"data before assigning to state dict =  {state_dict[name].dtype}")
        param.requires_grad = False
        #param.dtype = torch.int8
        state_dict[name] = state_dict[name].to(torch.int8)
        param.data = quantized_tensor
        state_dict[name].copy_(param)
        #print(str(param.data.dtype) + " right now")
        #print(f"data type after assigning to state dict: {state_dict[name].dtype}")
        #print(f"data type after assigning to state dict: {param.dtype}")
        #setattr(name + "_maxval", maxval)

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
print("at this point i've replaced the model parameters with the quantized parameters")

# poor man's data loader
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

finalLogits = None
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model.forward_quantize(X, Y, max_vals=max_vals)
                finalLogits = logits
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

losses = estimate_loss()
print(losses)

#code the calculation to the perplexity score now



