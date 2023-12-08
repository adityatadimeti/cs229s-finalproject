
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

device = "cpu"
ckpt_path = os.path.join("pretrained", 'ckpt.pt')
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


temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
init_from="gpt2"
max_new_tokens = 10 # number of tokens generated in each sample
num_samples = 1 # number of samples to draw

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


""" model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  """

# look for the meta pickle in case it is available in the dataset folder
print("No meta.pkl found, assuming GPT-2 encodings...")
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

start = "First Citizen: We are accounted poor citizens, the patricians good. What authority surfeits on would relieve us: if they would yield us but the superfluity, while it were wholesome, we might guess they relieved us humanely; but they think we are too dear: the leanness that afflicts us, the object of our misery, is as an inventory to particularise their abundance; our sufferance is a gain to them Let us revenge this with our pikes, ere we become rakes: for the gods know I speak this in hunger for bread, not in thirst for revenge. Second Citizen: Would you proceed especially against Caius Marcius? All: Against him first: hes a very dog to the commonalty. Second Citizen: Consider you what services he has done for his country? First Citizen: Very well; and could be content to give him good report fort, but that he pays himself with being proud. Second Citizen: Nay, but speak not maliciously. First Citizen: I say unto you, what he hath done famously, he did it to that end: though soft-conscienced men can be"
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])


#WITHOUT QUANTIZATION
print("WITHOUT QUANTIZATION")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=False)
            print(decode(y[0].tolist()))
            print('---------------')





#WITHOUT QUANTIZATION
print("WITH QUANTIZATION")

def perplexity_score(probs: np.ndarray) -> float:
    """Compute perplexity score from softmax probabilities."""
    return np.exp(-np.mean(np.log(probs)))

def quantize_param(param): #this data is the param data
    
    abs_max = torch.max(torch.abs(param))
    scaled_tensor = param / abs_max
    scaled_tensor = (scaled_tensor * 127.)

    clipped_tensor = torch.clamp(scaled_tensor, -127., 127.)
    quantized_tensor = clipped_tensor.to(torch.int8)
    quantized_tensor = quantized_tensor.to(torch.int8)
    return quantized_tensor, abs_max
    

def dequantize_param(param, maxval):
    """ rangemax = 127
    dequantized_tensor = param.to(torch.float16)
    dequantized_tensor = dequantized_tensor / rangemax * maxval
    return dequantized_tensor """

    rangemax = 127
    dequantized_tensor = param.to(torch.float32)
    dequantized_tensor = dequantized_tensor / rangemax * maxval
    dequantized_tensor = dequantized_tensor.to(torch.float32)
    return dequantized_tensor

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
        #print(f"data type after assigning to state dict: {state_dict[name].dtype}")
        #print(f"data type after assigning to state dict: {param.dtype}")
        #setattr(name + "_maxval", maxval)

print("at this point i've replaced the model parameters with the quantized parameters")


#run inference with this quantized model
print("generating on the quantized model")
with torch.no_grad():
    with ctx:
        #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=True, max_vals=max_vals)
        print(decode(y[0].tolist()))
        print('---------------')












print("3 and 4 stuff")

#PART 3 AND 4


#STANDARD DECODING WITH M
device = "cpu"
ckpt_path = os.path.join("pretrained-medium", 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']

gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.to(device)
state_dict = checkpoint['model']
modelCopy = model

unwanted_prefix = '_orig_mod.'

for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

print("STANDARD DECODING WITH M")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=False)
            print(decode(y[0].tolist()))
            print('---------------')

#SPEC DECODING WHERE D IS THE DRAFT MODEL, D = GPT2

print('Loading draft model')
draft_model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
"""
draft_model.eval()
draft_model.to(device) """

generations_spec = []
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            torch.manual_seed(k+1337) # we want consistency
            y = model.generate_speculative(x, max_new_tokens, draft_model, temperature=temperature, top_k=top_k)
            generations_spec.append(y[0].tolist())
            print(decode(generations_spec[-1]))
            print('---------------')

#quantize M now part 4

def quantize_param(param): #this data is the param data
    
    abs_max = torch.max(torch.abs(param))
    scaled_tensor = param / abs_max
    scaled_tensor = (scaled_tensor * 127.)

    clipped_tensor = torch.clamp(scaled_tensor, -127., 127.)
    quantized_tensor = clipped_tensor.to(torch.int8)
    quantized_tensor = quantized_tensor.to(torch.int8)
    return quantized_tensor, abs_max
    

def dequantize_param(param, maxval):
    """ rangemax = 127
    dequantized_tensor = param.to(torch.float16)
    dequantized_tensor = dequantized_tensor / rangemax * maxval
    return dequantized_tensor """

    rangemax = 127
    dequantized_tensor = param.to(torch.float32)
    dequantized_tensor = dequantized_tensor / rangemax * maxval
    dequantized_tensor = dequantized_tensor.to(torch.float32)
    return dequantized_tensor

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
        #print(f"data type after assigning to state dict: {state_dict[name].dtype}")
        #print(f"data type after assigning to state dict: {param.dtype}")
        #setattr(name + "_maxval", maxval)

print("at this model M, which is main model is now quantized")
generations_spec_quant = []
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            torch.manual_seed(k+1337) # we want consistency
            y = modelCopy.generate_speculative(x, max_new_tokens, model, temperature=temperature, top_k=top_k, max_vals=max_vals)
            generations_spec_quant.append(y[0].tolist())
            print(decode(generations_spec_quant[-1]))
            print('---------------')