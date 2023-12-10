import time

 # This is where the checkpoints will be saved
 # Take note to make sure saved checkpoints don't override each other, depending on what you are trying to accomplish (e.g. for pruning, may want to keep a few checkpoints around)
out_dir = 'shakespeare'
eval_interval = 10
eval_iters = 50

# logging
#log_interval = 10
log_interval = 1
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 8
gradient_accumulation_steps = 40
max_iters =

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

#pruning purposes
prune_at = 100
prune_type = "None"
prune_percentile = 0.9
