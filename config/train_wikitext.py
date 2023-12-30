# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = False
wandb_project = 'wikitext'
wandb_run_name='gpt2-124M'

# checkpoint
out_dir = 'wikitext'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 8
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 50
eval_iters = 100
log_interval = 10

# weight decay
weight_decay = 1e-1
# # config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# # launch as the following (e.g. in a screen session) and wait ~5 days:
# # $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# wandb_log = True
# wandb_project = 'wikitext'
# wandb_run_name='gpt2-124M'

# # checkpoint
# out_dir = 'wikitext'

# # these make the total batch size be ~0.5M
# # 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
# batch_size = 1
# block_size = 1024
# gradient_accumulation_steps = 5 * 8

# # this makes total number of tokens be 300B
# max_iters = 1
# lr_decay_iters = 600000

# # eval stuff
# eval_interval = 1
# eval_iters = 1
# log_interval = 10

# # weight decay
# weight_decay = 1e-1
