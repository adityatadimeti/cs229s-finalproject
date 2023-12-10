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

prune_types = ["zero_indiv", "zero_row", "reduce", "None"]
prune_percentiles = [0.9, 0.5, 0.1]
train_batches = [8, 12, 4]
dataset = wikitext
out_dir = f"pruned_models/{dataset}"
cur_params = {"config_file": f"config/train_{dataset}.py", "eval_iters": 10, "log_interval": 1, "max_iters": 500, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 8, "prune_type": "None", "prune_at": 100, "prune_percentile": 0.9, "out_dir": out_dir}

# All Models
prune_at = 100
model_dirs = {}
for batch_size in train_batches:
    cur_model_dirs = []

    # Baseline Model
    cur_out_dir = out_dir + "_" + prune_type + "_" + str(batch_size)
    cur_params['batch_size'] = batch_size
    cur_params['prune_type'] = "None"
    experiment_train(cur_params)
    cur_model_dirs.append(cur_out_dir)
    for prune_percentile in prune_percentiles:
        for prune_type in prune_types:
            cur_out_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_" + str(batch_size)
            cur_params['out_dir'] = cur_out_dir
            cur_params['prune_type'] = prune_type
            cur_params['prune_at'] = prune_at
            cur_params['prune_percentile'] = prune_percentile
            cur_params['batch_size'] = batch_size
            if batch_size != 8 or prune_percentile != 0.9:
                cur_params['max_iters'] = 150
            else:
                cur_params['max_iters'] = 500
            experiment_train(cur_params)
            cur_model_dirs.append(cur_out_dir)
    model_dirs[batch_size] = cur_model_dirs

"DONE WITH TRAINING"
print(model_dirs)

model_results = {}
for prune_type in prune_types:
    if prune_type == "None":
        cur_model = prune_type
        print("LOADING RESULTS ON ", cur_model)
        model_results[cur_model] = {}
        model_8_dir = out_dir + "_" + prune_type + "_8/ckpt.pt"
        model_12_dir = out_dir + "_" + prune_type + "_12/ckpt.pt"
        model_4_dir = out_dir + "_" + prune_type + "_4/ckpt.pt"

        model_8 = torch.load(model_8_dir, map_location=device)
        model_12 = torch.load(model_12_dir, map_location=device)
        model_4 = torch.load(model_4_dir, map_location=device)
        print(" 8-batch", model_8)
        print(" 4-batch", model_4)
        print(" 12-batch", model_12)
        model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
        model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
        model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]

    else:
        for prune_percentile in prune_percentiles:
            print("LOADING RESULTS ON ", cur_model)
            cur_model = prune_type + "_" + prune_percentile
            model_results[cur_model] = {}
            model_8_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_8/ckpt.pt"
            model_4_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_4/ckpt.pt"
            model_12_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_12/ckpt.pt"

            model_8 = torch.load(model_8_dir, map_location=device)
            model_12 = torch.load(model_12_dir, map_location=device)
            model_4 = torch.load(model_4_dir, map_location=device)

            print(" 8-batch", model_8)
            print(" 4-batch", model_4)
            print(" 12-batch", model_12)
            model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
            model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
            model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]


with open("run_all.json", "w") as outfile:
     json.dump(model_results, outfile)
