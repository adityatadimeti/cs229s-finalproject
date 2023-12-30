
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
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
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
start = "sample.jsonl" 


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
# def zero_point_quantize_param(param):
#     rangemin = -128.
#     rangemax = 127.

#     max = torch.max(param)
#     min = torch.min(param)
    
#     scale_factor = (rangemax-rangemin) / (max-min)
#     quantized_tensor = scale_factor * param
    
#     #offset to using the zeroopoint
#     zero_point = (-1 * scale_factor * min)
#     zero_point += rangemin
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
#IN THIS FILE, IT IS BEST TO JUST RUN ONE OF THE EXPERIMENTS AT A TIME INSTEAD OF UNCOMMENTING EVERYTHING
#AND RUNNING IT ALL AT ONCE, SO YOU CAN DISTINGUISH THE RESULTS FOR EACH ONE AND ALSO PREVENT ANY DOWNSTREAM 
#ERRORS FROM OCCURING THAT STALLS OTHER EXPERIMENTS
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

modelsize = 0
for param in model.parameters():
    modelsize += param.nelement() * param.element_size()
for buffer in model.buffers():
    modelsize += buffer.nelement() * buffer.element_size()

print("model size in bytes " + str(modelsize))

batch_size = 4 #change /toggle this batch size from 4 to 12

print("without quantization loss and perplexity, before inference")
losses = estimate_loss(isQuantize=False)
print(losses)

perplexity = torch.exp(losses['val'])
print("perplex val " + str(perplexity))

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

prompt_length = 1024
batch_size = 1

#lines=["once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit.", "once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit.", "once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit.", "once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit.", "once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit."]
#test on just one prompt for now to make it run faster
lines=["once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit."]
start_ids = [encode(x) for x in lines]
print(len(start_ids[0]))
print('NUM SATISFYING PROMPTS =', len([x for x in start_ids if len(x) > prompt_length]))
x = torch.tensor([x[:prompt_length] for x in start_ids if len(x) > prompt_length][:batch_size], dtype=torch.long, device=device)
print('RUNNING WITH BATCH SIZE =', x.size(0))

num_samples=1
max_new_tokens=10
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=False)
            print(decode(y[0].tolist()))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency))

print("without quantization loss and perplexity, after inference")
losses = estimate_loss(isQuantize=False)
print(losses)

perplexity = torch.exp(losses['val'])
print("perplex val " + str(perplexity))


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

print("with quantization loss and perplexity before inference")
losses = estimate_loss(isQuantize=True, max_vals=max_vals)
print(losses)

perplexity = torch.exp(losses['val'])
print("perplex val " + str(perplexity))

modelsize = 0
for param in model.parameters():
    modelsize += param.nelement() * param.element_size()
for buffer in model.buffers():
    modelsize += buffer.nelement() * buffer.element_size()

print("model size in bytes quantized " + str(modelsize))


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



max_new_tokens=50
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=4, isQuantize=True, max_vals=max_vals)
            print(decode(y[0].tolist()))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency))

print("with quantization loss and perplexity, after inference")
losses = estimate_loss(isQuantize=True, max_vals=max_vals)
print(losses)











print("3 and 4 stuff")

# #PART 3 AND 4
batch_size=1
print('Loading main model')
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

prompt_length = 1024
batch_size = 1

lines=["once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit."]
start_ids = [encode(x) for x in lines]
print('NUM SATISFYING PROMPTS =', len([x for x in start_ids if len(x) > prompt_length]))
x = torch.tensor([x[:prompt_length] for x in start_ids if len(x) > prompt_length][:batch_size], dtype=torch.long, device=device)
print('RUNNING WITH BATCH SIZE =', x.size(0))

max_new_tokens=50
print("STANDARD DECODING WITH M")
t = time.time()
num_samples=1
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = mainmodel.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=False)
            print(decode(y[0].tolist()))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency))

#SPEC DECODING WHERE D IS THE DRAFT MODEL, D = GPT2
print("Now comparing with spec decoding with M and D")
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

prompt_length = 960 #reduce this by four so that spec decoding generates the 4 num_speculative tokens
batch_size = 1

lines=["once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit."]
start_ids = [encode(x) for x in lines]
print(len(start_ids[0]))
print('NUM SATISFYING PROMPTS =', len([x for x in start_ids if len(x) > prompt_length]))
x = torch.tensor([x[:prompt_length] for x in start_ids if len(x) > prompt_length][:batch_size], dtype=torch.long, device=device)
print('RUNNING WITH BATCH SIZE =', x.size(0))

max_new_tokens=50
generations_spec = []
print("starting generation")
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(1):
            print(" next sample")
            torch.manual_seed(k+1337) # we want consistency
            print(x.size())
            y = mainmodel.generate_speculative(x, max_new_tokens, draftmodel, temperature=temperature, top_k=top_k)
            generations_spec.append(y[0].tolist())
            print(decode(generations_spec[-1]))
            print('---------------')

timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency))

#quantize M now part 4
print("make M quantized and do standard decoding")
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
prompt_length = 960 #reduce this by four so that spec decoding generates the 4 num_speculative tokens
batch_size = 1

lines=["once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit."]
start_ids = [encode(x) for x in lines]
print('NUM SATISFYING PROMPTS =', len([x for x in start_ids if len(x) > prompt_length]))
x = torch.tensor([x[:prompt_length] for x in start_ids if len(x) > prompt_length][:batch_size], dtype=torch.long, device=device)
print('RUNNING WITH BATCH SIZE =', x.size(0))
print("STANDARD DECODING WITH M quantized")
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(1):
            #y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            y = mainmodel.generate(x, max_new_tokens, temperature=temperature, top_k=top_k, isQuantize=True, max_vals=max_vals)
            print(decode(y[0].tolist()))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency))


print('Loading main model')
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
prompt_length = 960 #reduce this by four so that spec decoding generates the 4 num_speculative tokens
batch_size = 1

lines=["once upon a time in a quaint little village nestled between rolling hills and meandering streams, there lived a young dreamer named lila. she spent her days exploring the enchanting woods that surrounded her home, where ancient trees whispered secrets and magical creatures danced in the dappled sunlight. lila's heart brimmed with curiosity, and her eyes sparkled with a relentless thirst for adventure. one fateful day, while chasing butterflies near the edge of the forest, she stumbled upon a hidden portal. drawn by an unseen force, lila stepped through the shimmering veil, finding herself in a realm of wonder and mystery. the new world was a kaleidoscope of colors and sounds, with floating islands suspended in the sky and waterfalls cascading upwards. lila marveled at the beauty that unfolded before her, a symphony of nature's wonders. as she explored this fantastical realm, she encountered a wise old owl named elara, who revealed that lila was the chosen one destined to restore balance to the fractured land. guided by ancient prophecies, lila embarked on a quest to retrieve the lost shards of a powerful crystal that held the key to harmony. her journey led her through dense jungles, across vast deserts, and into the heart of a mystical city where time itself seemed to dance. along the way, lila forged unlikely alliances with a mischievous sprite, a gentle giant, and a clever fox who possessed the gift of prophecy. together, they faced trials and tribulations, overcoming formidable foes and solving riddles that guarded the precious crystal shards. as lila grew in strength and wisdom, so did the bond between her and her newfound companions. their friendship became a beacon of hope in the face of darkness, inspiring the inhabitants of the fractured land to unite against the forces that sought to sow discord. through moments of laughter and tears, lila discovered the true power of the human spirit and the magic that lies within the hearts of those who dare to dream. as the final shard was placed in its rightful position, the land began to heal, and the once-divided realms merged into a tapestry of unity and peace. lila, now a heroine celebrated by all, returned to her village, forever changed by the extraordinary adventure that had unfolded. the villagers, too, were transformed, inspired by lila's courage to embrace the magic that exists in every corner of the world. and so, the once quaint village became a hub of creativity and innovation, a testament to the enduring power of dreams and the boundless possibilities that lie beyond the veil of the ordinary. the seasons changed, and lila's village blossomed into a thriving community of artists, inventors, and dreamers who celebrated the magic of life. inspired by lila's tales, they built bridges of understanding with neighboring villages, creating a network of harmony that stretched across the lands. festivals filled with laughter and song became a regular occurrence, uniting people from all walks of life. lila, now a revered elder, continued to share her wisdom and the lessons learned from her extraordinary journey. the village, once humble, now stood as a beacon of inspiration for those who dared to dream beyond the confines of the ordinary. yet, as the years passed, a shadow loomed on the horizon. rumors of a looming darkness began to circulate, whispers of an ancient evil awakening from a dormant slumber. lila, sensing the impending threat, gathered a new generation of adventurers, each inheriting the spirit of courage and curiosity. among them was a spirited young artist named ari, who possessed a unique ability to paint visions of the future. together, this new fellowship embarked on a quest to uncover the mysteries of the ancient evil and prevent the impending doom that threatened to shroud their world in darkness. their journey took them to forgotten temples, where cryptic inscriptions revealed the history of an age-old conflict between light and shadow. they encountered mythical beings, each holding a piece of the puzzle that could unlock the secrets of the impending darkness. ari's paintings guided them through treacherous landscapes and hidden realms, offering glimpses into the possible futures that awaited them. as the fellowship delved deeper into the heart of the mystery, they discovered a connection between the ancient evil and the very crystal shards lila had once sought to restore harmony. a grand prophecy unfolded, foretelling of a chosen one who would rise to confront the darkness and, once again, bring balance to the world. ari, with her extraordinary visions, realized that the chosen one was none other than lila herself, reborn to face a new and formidable adversary. the fellowship, now guided by both lila and ari, pressed on with renewed determination, knowing that the fate of their world rested on their shoulders. the journey led them to a desolate wasteland, where the air was thick with despair and the very essence of magic seemed to wither away. here, they confronted the ancient evil, a malevolent force that sought to devour the light and plunge the world into eternal night. the battle that ensued tested the limits of their strength and resolve, but the fellowship stood united, drawing upon the lessons learned from lila's original quest. in the midst of the epic confrontation, ari's paintings began to weave themselves into reality, creating portals that allowed glimpses of hope and inspiration to pour into the battlefield. the ancient evil, confronted by the unyielding power of creativity and the indomitable human spirit, began to waver. sensing an opportunity, lila stepped forward, wielding the crystal shards that had once restored harmony to the fractured land. a surge of energy enveloped lila, transforming her into a radiant beacon of light. with unwavering determination, she faced the ancient evil and unleashed the true potential of the crystal shards. a wave of brilliant energy cascaded across the battlefield, dispelling the darkness and restoring balance to the world once more. the fellowship, in awe of the spectacle before them, witnessed the birth of a new era. as the radiant glow faded, lila stood before them, her eyes filled with a profound sense of fulfillment. the ancient evil, now vanquished, left behind a legacy of lessons learned and a reminder that darkness could never truly extinguish the light. the fellowship, having played a crucial role in the battle, returned to their villages as heroes, carrying the wisdom gained from their perilous journey. lila, now an immortal guardian of the crystal shards, continued to watch over the world, ensuring that the lessons of unity and creativity endured through the ages. the once-forgotten villages became beacons of inspiration, and the memory of the fellowship's triumph lived on in the tapestry of history. ari, having discovered her own place in the grand design, continued to paint visions that inspired generations to come. the world, forever changed by the interplay of light and shadow, flourished in an era of enlightenment and creativity. the people, guided by the legacy of lila and the fellowship, embraced the magic within and forged a future where dreams knew no bounds. and so, the cycle of adventure and discovery continued, weaving a timeless tale of hope, courage, and the enduring power of the human spirit."]
start_ids = [encode(x) for x in lines]
print('NUM SATISFYING PROMPTS =', len([x for x in start_ids if len(x) > prompt_length]))
x = torch.tensor([x[:prompt_length] for x in start_ids if len(x) > prompt_length][:batch_size], dtype=torch.long, device=device)
print('RUNNING WITH BATCH SIZE =', x.size(0))
num_samples = 3
print("at this point the main model is also now quantized")
generations_spec_quant = []
print("starting generation")
t = time.time()
with torch.no_grad():
    with ctx:
        for k in range(1):
            print(" next sample")
            torch.manual_seed(k+1337) # we want consistency
            y = mainmodel.generate_speculative(x, max_new_tokens, quantizedraft, temperature=temperature, top_k=top_k, max_vals=max_vals, isMainModelQuantize=False, isDraftModelQuantize=True)
            generations_spec_quant.append(y[0].tolist())
            print(decode(generations_spec_quant[-1]))
            print('---------------')
timeComplete = time.time() - t
latency = timeComplete / num_samples
print("time completed " + str(timeComplete))
print("latency " + str(latency))