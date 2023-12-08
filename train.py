

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

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

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
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
dataset = 'wikitext'
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
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
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

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# # training loop
# X, Y = get_batch('train') # fetch the very first batch
# t0 = time.time()
# local_iter_num = 0 # number of iterations in the lifetime of this process
# raw_model = model.module if ddp else model # unwrap DDP container if needed
# running_mfu = -1.0

# # import torch.nn.utils.prune as prune

# #prune.l1_unstructured(module, name="bias", amount=3)


# """
# overall: take top 10% of weights ACROSS ALL parameters, not just individual parameters. certain parameters are weights and certain ones are biases. 
# Step by step:
# 1 - identify parameters that are weight parameters
# 2 - coalesce all the weight parameters via flatten, which puts them into a single list
# 3 - reverse - sort them and identify the weight value that's at the 10% mark
# 4 - then you develop a mask, per weight parameter (i.e. for each parameter thats a weight), 
#     that is 1 if the weight is above the 10% mark, and 0 if it's below the 10% mark. 
#     note that this mask is fixed for each weight parameter, so the mask never changes. 
# 5 - you repeatedly multiply this mask with the weight parameter since there are occasisins where the weight
#     parameter is 0 and then becomes nonzero because it gets learned. we want to prevent this. 
# """

# mask_dict = {} # moving this outside while training loop so mask_dict is not reinitlized for each iteration
# for name, param in model.named_parameters():
#     mask_dict[name] = torch.ones_like(param) # note that we can use name as a key because each name is unique per layer/type
#     #print(name, mask_dict[name])

# # Calculate the percentage of data values across all entries of all weight matrices across all parameters that equal 0
# # total_values = 0
# # zero_values = 0

# # for name, param in model.named_parameters():
# #     if "weight" in name:
# #         total_values += param.numel()
# #         zero_values += (param == 0).sum().item()

# # percent_zero_values = (zero_values / total_values) * 100
# #print("Percentage of weight values that equal 0:", percent_zero_values)

# file_path = 'prune_data_'+str(eval_iters)+'_'+str(batch_size)+'.pickle'


# while True:
#     print("INSIDE ITERATION " , iter_num)
#     # determine and set the learning rate for this iteration
#     lr = get_lr(iter_num) if decay_lr else learning_rate
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
   
#     if (iter_num == 100):
#         print("time to prune")
#         # create a lambda function that flattens all parameters that have "weight" in name
#         fn = file_path
#         if os.path.isfile(fn):
#             print("pckle file exists")
#             with open(fn, 'rb') as f:
#                 prune_val = pickle.load(f)
#                 iter_num = pickle.load(f)
#                 #flattened_parameters = pickle.load(f)
#                 #sorted_list = pickle.load(f)
#                 mask_dict = pickle.load(f)
#                 batch_size = pickle.load(f)

#                 for name, param in model.named_parameters():
#                     if "weight" in name:
#                         # this is a weight parameter
#                         mask_dict[name][torch.abs(param.data) < prune_val] = 0
#                         print(name, mask_dict[name])
#                         with torch.no_grad(): # freezes the grad for the entire matrix??? or just the mask?
#                             param *= mask_dict[name]
        
#         else:
#             print("no pickle file, so running manual process")

#             layer_weights_count = 0
#             flattened_tensors = []
#             for name, param in model.named_parameters():
#                 if 'weight' in name:
#                     flattened_tensors.append(param.data.flatten())
#                     layer_weights_count += param.data.flatten().numel()
#                     print(layer_weights_count)
#             flattened_weights = torch.cat(flattened_tensors)
#             sorted_indices = torch.argsort(torch.abs(flattened_weights), descending=True)
#             sorted_flattened_weights = flattened_weights[sorted_indices]
#             #print(sorted_flattened_weights)
#             threshold_index = int(layer_weights_count/10 + 1)
#             #print(sorted_flattened_weights[threshold_index])
#             prune_val = abs(sorted_flattened_weights[threshold_index])

#             # flatten_parameters = lambda: [param.data.flatten() for name, param in model.named_parameters() if "weight" in name]

#             # # flatten the parameters and sort them by their magnitudes
#             # flattened_parameters = flatten_parameters() # this is a list of tensors
#             # flattened_and_concatenated = [tensor.tolist() for tensor in flattened_parameters]
#             # flattened_and_concatenated = [item for sublist in flattened_and_concatenated for item in sublist]
#             # sorted_list = sorted(flattened_and_concatenated, key=abs, reverse=True)

#             # find the index to prune up to
#             # prune_index = int(len(sorted_list) * 0.1)
#             # prune_val = sorted_list[prune_index]
#             print("done with evaluations, starting to save")

#             #save this variable data to the pickle file
#             with open(file_path, 'wb') as f:
#                 pickle.dump(prune_val, f)
#                 pickle.dump(iter_num, f)
#                 #pickle.dump(flattened_parameters, f)
#                 #pickle.dump(sorted_list, f)
#                 pickle.dump(mask_dict, f)
#                 pickle.dump(batch_size, f)
#             print("saved")

#             for name, param in model.named_parameters():
#                 if "weight" in name:
#                     # this is a weight parameter
#                     #print("sum of weights before pruning", torch.sum(torch.abs(param.data)))
#                     mask_dict[name][torch.abs(param.data) < prune_val] = 0
#                     #print(name, mask_dict[name])
#                     with torch.no_grad(): # freezes the grad for the entire matrix??? or just the mask?
#                         param *= mask_dict[name]
#         # prune_val = 0.4 # hard coding this for debugging purposes

        
        
#         #print(len(sorted_list), prune_val, max(sorted_list), min(sorted_list))
    
#     else:
#         for name, param in model.named_parameters():
#             if "weight" in name:
#                 with torch.no_grad(): # freezes the grad for the entire matrix??? or just the mask?
#                     param *= mask_dict[name]

        

    
#     # Calculate the percentage of data values across all entries of all weight matrices across all parameters that equal 0
#     # total_values = 0
#     # zero_values = 0

#     # for name, param in raw_model.named_parameters():
#     #     if "weight" in name:
#     #         total_values += param.numel()
#     #         zero_values += (param == 0).sum().item()

#     # percent_zero_values = (zero_values / total_values) * 100
#     # print("Percentage of weight values that equal 0:", percent_zero_values)

#     # evaluate the loss on train/val sets and write checkpoints
#     if iter_num % eval_interval == 0 and master_process:
#         losses = estimate_loss()
#         print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#         if wandb_log:
#             wandb.log({
#                 "iter": iter_num,
#                 "train/loss": losses['train'],
#                 "val/loss": losses['val'],
#                 "lr": lr,
#                 "mfu": running_mfu*100, # convert to percentage
#             })
#         if losses['val'] < best_val_loss or always_save_checkpoint:
#             best_val_loss = losses['val']
#             if iter_num > 0:
#                 checkpoint = {
#                     'model': raw_model.state_dict(),
#                     'optimizer': optimizer.state_dict(),
#                     'model_args': model_args,
#                     'iter_num': iter_num,
#                     'best_val_loss': best_val_loss,
#                     'config': config,
#                 }
#                 print(f"saving checkpoint to {out_dir}")
#                 torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
#                 print(f"done saving")
#     if iter_num == 0 and eval_only:
#         break

#     # forward backward update, with optional gradient accumulation to simulate larger batch size
#     # and using the GradScaler if data type is float16
#     for micro_step in range(gradient_accumulation_steps):
#         if ddp:
#             # in DDP training we only need to sync gradients at the last micro step.
#             # the official way to do this is with model.no_sync() context manager, but
#             # I really dislike that this bloats the code and forces us to repeat code
#             # looking at the source of that context manager, it just toggles this variable
#             model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
#         with ctx:
#             logits, loss = model(X, Y)
#             loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
#         # immediately async prefetch next batch while model is doing the forward pass on the GPU
#         X, Y = get_batch('train')
#         # backward pass, with gradient scaling if training in fp16
#         scaler.scale(loss).backward()
#     # clip the gradient
#     if grad_clip != 0.0:
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
#     # step the optimizer and scaler if training in fp16
#     scaler.step(optimizer)
#     scaler.update()
#     # flush the gradients as soon as we can, no need for this memory anymore
#     optimizer.zero_grad(set_to_none=True)

#     # timing and logging
#     t1 = time.time()
#     dt = t1 - t0
#     t0 = t1
#     if iter_num % log_interval == 0 and master_process:
#         # get loss as float. note: this is a CPU-GPU sync point
#         # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
#         lossf = loss.item() * gradient_accumulation_steps
#         if local_iter_num >= 5: # let the training loop settle a bit
#             mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
#             running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
#         print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
#     iter_num += 1
#     local_iter_num += 1

#     # termination conditions
#     if iter_num > max_iters:
#         break

"""
You cannot use third party or built in quantization libraries. You are expected to implement the quantization part using only native PyTorch code. We only expect you to implement simulated quantization where you still compress the model weights into their quantized form to reduce the overall memory usage, but you can dequantize the weights on-the-fly for the actual inference computation. We also only require post-training quantization

In this part, you are tasked with implementing 8-bit quantized inference, post-training quantization, with the GPT-2 pretrained model. Please quantize the model weights. 
"""

if ddp:
    destroy_process_group()



#this is outside, at the end, of the training file

def quantize_param(param): #this data is the param data
    rangemin = -128
    rangemax = 127

    maxval = torch.max(torch.abs(param))

    #scaled tensor
    scaling_factor = rangemax / maxval
    scaled_tensor = param * scaling_factor

    clipped_tensor = torch.clamp(scaled_tensor, rangemin, rangemax)
    quantized_tensor = torch.round(clipped_tensor)
    quantized_tensor = quantized_tensor.to(torch.int8)
    quantized_tensor.requires_grad = False
    return quantized_tensor, maxval
    

def dequantize_param(param, maxval):
    rangemax = 127
    dequantized_tensor = param.to(torch.float16)
    dequantized_tensor = dequantized_tensor / rangemax * maxval
    return dequantized_tensor


if init_from == 'resume':
    # init from a model saved in a specific directory

    print("resuming from model from " + out_dir)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

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

print("quantizing the model parameters")
max_vals = {} #this is a dictionary that maps the name of the parameter to the max value of that parameter
print("param names")
for name, param in model.named_parameters():
    if ".wte." in name or ".wpe." in name or "weight" in name or "bias" in name:
        print(name)
        quantized_tensor, maxval = quantize_param(param.data)
        #print(name + "_maxval")
        max_vals[name+"_maxval"] = maxval
        #assign the quantized tensor back to the model parameter
        print(f"data before assigning to state dict =  {state_dict[name].dtype}")
        param.requires_grad = False
        #param.dtype = torch.int8
        state_dict[name] = state_dict[name].to(torch.int8)
        param.data = quantized_tensor
        state_dict[name].copy_(param)
        print(f"data type after assigning to state dict: {state_dict[name].dtype}")
        #print(f"data type after assigning to state dict: {param.dtype}")
        #setattr(name + "_maxval", maxval)


print("at this point i've replaced the model parameters with the quantized parameters")


print("loading params for inference")
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'shakespeare' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


start = "First Citizen: We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might guess they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them Let us revenge this with our pikes, ere we become rakes: for the gods know I speak this in hunger for bread, not in thirst for revenge. Second Citizen: Would you proceed especially against Caius Marcius? All: Against him first: hes a very dog to the commonalty. Second Citizen: Consider you what services he has done for his country? First Citizen: Very well; and could be content to give him good report fort, but that he pays himself with being proud. Second Citizen: Nay, but speak not maliciously. First Citizen: I say unto you, what he hath done famously, he did it to that end: though soft-conscienced men can be"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


#max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.8
top_k = 200

#run inference with this quantized model
print("generating on the quantized model")
with torch.no_grad():
    with ctx:
        #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=True, max_vals=max_vals)
        print(decode(y[0].tolist()))
        print('---------------')



#part 3
##speculative decoding and onwards is here

#run speculative decoding on the model
""" generations_spec = []
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            torch.manual_seed(k+1337) # we want consistency
            y = model.generate_speculative(x, max_new_tokens, draft_model, temperature=temperature, top_k=top_k)
            generations_spec.append(y[0].tolist())
            print(decode(generations_spec[-1]))
            print('---------------') """

#num_speculative is always 4



