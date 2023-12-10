
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
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
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
num_samples = 5 # number of samples to draw
dataset = 'wikitext'
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
start = "\n" 


def quantize_param(param): #this data is the param data
    rangemin = -128.
    rangemax = 127.
    
    maxval = torch.max(torch.abs(param))
    scaled_tensor = param / maxval * rangemax

    quantized_tensor = torch.clamp(scaled_tensor, rangemin, rangemax)
    quantized_tensor = torch.round(quantized_tensor)
    quantized_tensor = quantized_tensor.to(torch.int8)

    quantized_tensor.requires_grad = False
    return quantized_tensor, maxval

def dequantize_param(param, maxval):
    rangemax = 127.
    dequantized_tensor = param.to(torch.float32)
    dequantized_tensor = dequantized_tensor / rangemax * maxval
    return dequantized_tensor

#this is the zero point quantization version that we tested with but didnt find that it performed any better than the absolute max, nor the naive implementation
# def zero_point_quant(param):
#     rangemin = -128.
#     rangemax = 127.

#     max = torch.max(param)
#     min = torch.min(param)
    
#     scale_factor = (rangemax-rangemin) / (max-min)
#     quantized_tensor = scale_factor * param
    
#     #offset to using the zeroopoint
#     zero_point = (-1 * scale_factor * min) + rangemin
#     quantized_tensor = quantized_tensor + zero_point

#     quantized_tensor = torch.clamp(quantized_tensor, rangemin, rangemax)
#     quantized_tensor = torch.round(quantized_tensor)
#     quantized_tensor = quantized_tensor.to(torch.int8)
#     return quantized_tensor, scale_factor, zero_point


# poor man's data loader
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    print("getting batch " + str(batch_size))
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

@torch.no_grad()
def estimate_loss(isQuantize=False, max_vals=None, zerovals=None, scales=None):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if isQuantize:
                    logits, loss = model.forward_quantize(X, Y, max_vals=max_vals)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#BEST TO COMMENT OUT EITHER (PART 1 AND 2) OR (PART 3 AND 4) to run the code, best to run the code in two halves
#PART 1 AND 2 STUFF - CHANGE THE CHECKPOINT TO THE 12 BATCH SIZE TO GET THE OTHER RESULTS 
ckpt_path = os.path.join("pretrainedSmallBatchSize4", 'ckpt.pt') #use either pretrainedSmallBatchSize4 or pretrainedSmallBatchSize12 or pretrainedSmallBatchSize1 or pretrainedMediumBatchSize1 - all of these are 1024 seq len
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

batch_size = 12 #change /toggle this batch size from 4 to 12

print("without quantization loss and perplexity")
losses = estimate_loss(isQuantize=False)
print(losses)

perplexity = torch.exp(losses['val'])
print("perplex val " + str(perplexity))
memory_usage = torch.cuda.max_memory_allocated()

print("quantizing the model parameters")
max_vals = {} #this is a dictionary that maps the name of the parameter to the max value of that parameter
#zero_vals = {}
#scales = {}
mytest = None
for name, param in model.named_parameters():
    if ".wte." in name or ".wpe." in name or "weight" in name or "bias" in name:
        quantized_tensor, maxval = quantize_param(param.data)
        #quantized_tensor, scale, zero_point = zero_point_quant(param.data)
        #print(name + "_maxval")
        max_vals[name+"_maxval"] = maxval
        #zero_vals[name+"_zeroval"] = zero_point
        #scales[name+"_scale"] = scale
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

print("at this point i've replaced the model parameters with the quantized parameters")

print("with quantization loss and perplexity")
losses = estimate_loss(isQuantize=True, max_vals=max_vals)
print(losses)

perplexity = torch.exp(losses['val'])
print("perplex val " + str(perplexity))
print("mem usage" + str(torch.cuda.memory_allocated() ))












print("3 and 4 stuff")

# #PART 3 AND 4
batch_size=1
""" print('Loading main model')
ckpt_path = os.path.join("pretrainedMediumBatchSize1", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
mainmodel = GPT(gptconf)
mainmodel.to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

mainmodel.load_state_dict(state_dict)

load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]) """

# max_new_tokens=50
# print("STANDARD DECODING WITH M")
# t = time.time()
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             y = mainmodel.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=False)
#             print(decode(y[0].tolist()))
#             print('---------------')
# timeComplete = time.time() - t
# latency = timeComplete / num_samples
# print("time completed " + str(timeComplete))
# print("latency " + str(latency))

#SPEC DECODING WHERE D IS THE DRAFT MODEL, D = GPT2
""" print("Now comparing with spec decoding with M and D")
print('Loading draft model')
ckpt_path = os.path.join("pretrainedSmallBatchSize1", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
draftmodel = GPT(gptconf)
draftmodel.to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

draftmodel.load_state_dict(state_dict)

max_new_tokens=50
generations_spec = []
print("starting generation")
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(" next sample")
            torch.manual_seed(k+1337) # we want consistency
            y = mainmodel.generate_speculative(x, max_new_tokens, draftmodel, temperature=temperature, top_k=top_k)
            generations_spec.append(y[0].tolist())
            print(decode(generations_spec[-1]))
            print('---------------')

timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency)) """

#quantize M now part 4
""" print("make M quantized and do standard decoding")
print("load model")
ckpt_path = os.path.join("pretrainedMediumBatchSize1", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
mainmodel = GPT(gptconf)
mainmodel.to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

mainmodel.load_state_dict(state_dict)

print("quantizing the draft model parameters")
max_vals = {} #this is a dictionary that maps the name of the parameter to the max value of that parameter
print("param names")
for name, param in mainmodel.named_parameters():
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

max_new_tokens=50
print("STANDARD DECODING WITH M quantized")
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = mainmodel.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=True, max_vals=max_vals)
            print(decode(y[0].tolist()))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency)) """


""" print('Loading main model')
ckpt_path = os.path.join("pretrainedMediumBatchSize1", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
mainmodel = GPT(gptconf)
mainmodel.to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

mainmodel.load_state_dict(state_dict)


print('Loading draft model that is now going to be quantized')
ckpt_path = os.path.join("pretrainedMediumBatchSize1", 'ckpt.pt') #this model has already been finetuned, so just load it in
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
quantizedraft = GPT(gptconf)
quantizedraft.to(device)
state_dict = checkpoint['model']

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

quantizedraft.load_state_dict(state_dict)


print("quantizing the draft model parameters")
max_vals = {} #this is a dictionary that maps the name of the parameter to the max value of that parameter
print("param names")
for name, param in quantizedraft.named_parameters():
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

max_new_tokens=50
num_samples = 3
print("at this point the main model is also now quantized")
generations_spec_quant = []
print("starting generation")
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            print(" next sample")
            torch.manual_seed(k+1337) # we want consistency
            y = mainmodel.generate_speculative(x, max_new_tokens, quantizedraft, temperature=temperature, top_k=top_k, max_vals=max_vals, isMainModelQuantize=False, isDraftModelQuantize=True)
            generations_spec_quant.append(y[0].tolist())
            print(decode(generations_spec_quant[-1]))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency)) """