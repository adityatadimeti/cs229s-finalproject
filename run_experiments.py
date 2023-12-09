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
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
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
dataset = 'wikitext'
data_dir = os.path.join('data', dataset)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8

def get_batch(split):
    data  = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

#PART 1 VALIDATION LOSS ON WIKITEXT
ckpt_path = os.path.join("wikitext", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
trainTime = checkpoint['timeSeconds']

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.to(device)

t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=False)
            print(decode(y[0].tolist()))
            print('---------------')
timeCompletedInference = time.time() - t
tokens_generated = x.size(0) * num_samples * max_new_tokens
losses = estimate_loss()
totalTokens = num_iters * batch_size * block_size * gradient_accumulation_steps

print("VALIDATION LOSS STUFF " + losses)
print("INFERENCE LATENCY - TIME/SECONDS: " + timeCompletedInference)
print("INFERENCE LATENCY - TOKENS: " + totalTokens)
print("THROUGHPUT SAMPLES: " + train_num_samples)
print("THROUGHPUT TIME/SECONDS: " + trainTime)



#PART 2 AND 3 - INF LATENCY 
""" train_quality_command = f"python train.py config/train_shakespeare.py --device=cpu --compile=False --eval_iters=1 --log_interval=1 --block_size=64 --batch_size=8 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2 --lr_decay_iters=2000 --dropout=0.0 --gradient_accumulation_steps=40 --dataset=shakespeare"
time1 = time.time()
result = subprocess.run(train_quality_command, shell=True, capture_output=True, text=True)
time2 = time.time()
print(result.stdout) """

# @torch.no_grad()
# def estimate_inference_time(samples, batch_size):
#     t = time.time()
#     for i in range(samples):
#         X,Y = get_batch('train', batch_size)
#         with ctx:
#             _, loss = model(X,Y)
#     return samples/(time.time() - t)

# def estimate_train_time(samples, batch_size):
#     t = time.time()
#     for i in range(samples):
#         X,Y = get_batch('train', batch_size)
#         with ctx:
#             _, loss = model(X,Y)
#             loss.backward()
#     rate = samples/(time.time() - t)
#     optimizer.zero_grad(set_to_none=True)
#     return rate


# def experiment_train(params):
#     command = f"python train.py {params['config_file']} --eval_iters={params['eval_iters']} --log_interval={params['log_interval']}  --max_iters={params['max_iters']} --lr_decay_iters={params['lr_decay_iters']} --dropout={params['dropout']}"
#     print(command)
#     command = f"python train.py config/train_shakespeare.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2 --lr_decay_iters=2000 --dropout=0.0"
#     time1 = time.time()
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     time2 = time.time()
#     print(result.stdout)
#     return result, (time2 - time1)

# def experiment_test(params):
#     command = f"python sample.py --out_dir={params['out_dir']} --start={params['start']} --num_samples={params['num_samples']} --max_new_tokens={params['max_new_tokens']} --device=cpu"
#     time1 = time.time()
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     time2 = time.time()
#     return result

# #train_params_8 = {"config_file": "config/train_wikitext.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 8}
# #train_params_1 = {"config_file": "config/train_wikitext.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 1}
# #train_params_12 = {"config_file": "config/train_wikitext.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 12}
# #train_result_8, train_time_8 = experiment_train(train_params_8)

# test_params_1 = {"out_dir": "shakespeare", "start": "\'What is the answer to life, the universe, and everything?\'", "num_samples": 1, "max_new_tokens": 100}
# test_params_12 = {"out_dir": "shakespeare", "start": "\'What is the answer to life, the universe, and everything?\'", "num_samples": 12, "max_new_tokens": 100}
# test_result_1 = experiment_test(test_params_1)
# test_result_12 = experiment_test(test_params_12)

# #train_8_logs = str(train_result_8.stdout).split(" ")
# #last_occurrence_index = len(train_8_logs) - train_8_logs[::-1].index("val") - 1
# #val_loss = float(train_8_logs[last_occurrence_index+2])
# # train_samples
# #train_throughput = train_samples / train_time
# print(test_result_1.stdout)
# inference_latency_1 = float(str(test_result_1.stdout).split()[-2])
# inference_latency_12 = float(str(test_result_12.stdout).split()[-2])
# print(inference_latency_1, inference_latency_12)
# metric_dict = {"inference_latency_1": inference_latency_1, "inference_latency_12": inference_latency_12}
# with open("run.json", "w") as outfile:
#     json.dump(metric_dict, outfile)
