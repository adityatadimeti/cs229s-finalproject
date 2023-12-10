import subprocess
import ast
import json
import time

def experiment_train(params):
    prune_type = params['prune_type']
    prune_at = params['prune_at']
    prune_percentile = params['prune_percentile']
    command = f"python train.py {params['config_file']} --prune_tyle={prune_type} --prune_percentile={prune_percentile} --eval_iters={params['eval_iters']} --log_interval={params['log_interval']}  --max_iters={params['max_iters']} --lr_decay_iters={params['lr_decay_iters']} --dropout={params['dropout']} --out_dir={params['out_dir']}"
    print("Running,", prune_type, prune_percentile, prune_at)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print("Completed,", prune_type, prune_percentile, prune_at)

# def experiment_test(params):
#     command = f"python sample.py --out_dir={params['out_dir']} --start={params['start']} --num_samples={params['num_samples']} --max_new_tokens={params['max_new_tokens']} --device=cpu"
#     time1 = time.time()
#     result = subprocess.run(command, shell=True, capture_output=True, text=True)
#     time2 = time.time()
#     return result

dataset = wikitext
out_dir = f"pruned_models/{dataset}"
cur_params = {"config_file": f"config/train_{dataset}.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 8, "prune_type": "None", "prune_at": 100, "prune_percentile": 0.9, "out_dir": out_dir}

# All Models
prune_at = 100
model_dirs = {}
for batch_size in [8, 4, 12]:
    cur_model_dirs = []

    # Baseline Model
    cur_out_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_" + str(batch_size)
    cur_params['batch_size'] = batch_size
    cur_params['prune_type'] = "None"
    experiment_train(cur_params)
    cur_model_dirs.append(cur_out_dir)
    for prune_percentile in [0.9, 0.5, 0.1]:
        for prune_type in ["zero_indiv", "zero_row", "reduce"]:
            cur_out_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_" + str(batch_size)
            cur_params['out_dir'] = cur_out_dir
            cur_params['prune_type'] = prune_type
            cur_params['prune_at'] = prune_at
            cur_params['prune_percentile'] = prune_percentile
            cur_params['batch_size'] = batch_size
            experiment_train(cur_params)
            cur_model_dirs.append(cur_out_dir)
    model_dirs[batch_size] = cur_model_dirs

"DONE WITH TRAINING"
print(model_dirs)
model_results = {}
for size in model_dirs:
    for prune_percentile in [0.9, 0.5, 0.1]:
        for prune_type in ["zero_indiv", "zero_row", "reduce"]:
            cur_model = prune_type + "_" + str(prune_percentile)
            cur_model_file = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_" + str(batch_size) + "/ckpt.pt"
            if cur_model not in model_results:
                cur_checkpoint = torch.load(ckpt_path, map_location=device)
            print("Pulling train data from ", cur_model)
            cur_valid_loss = cur_checkpoint['val_loss']

    # load in training times
    # load in batch sizes
    # Compute tokens / second
    # load in valid loss
    # model_results[model_dir] = {valid, through_4}


# metric_dict = {"inference_latency_1": inference_latency_1, "inference_latency_12": inference_latency_12}
# with open("run.json", "w") as outfile:
#     json.dump(metric_dict, outfile)
