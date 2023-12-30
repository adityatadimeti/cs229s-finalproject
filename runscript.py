import os
import time
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

model_8 = torch.load('wikitext/ckpt.pt', map_location='cpu')
print(model_8)
print("pruned", model_8['percent_nonzero'])
print("throughput", model_8['iter_num']*model_8['tokens_per_iter']/model_8['total_time'])
print("valid_loss", model_8['best_val_loss'].item())