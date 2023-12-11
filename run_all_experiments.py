import subprocess
import ast
import json
import time
import torch
import sys

#command = "python train.py config/train_wikitext.py --compile=False --log_interval=1 --block_size=64 --batch_size=8 --n_layer=4 --n_head=4 --n_embd=128 --prune_type=reduce --prune_at=100 --max_iters=500 —lr_decay_iters=500 --outdir=prune_models/wikitext_reduce_100_0.9_8 --dropout=0.0"
# command = "python train.py config/train_wikitext.py --compile=False --log_interval=1 --block_size=64 --batch_size=8 --n_layer=4 --n_head=4 --n_embd=128 --prune_type=reduce --prune_at=2 --max_iters=3 —-lr_decay_iters=3 --outdir=prune_models/wikitext_reduce_100_0.9_8 --dropout=0.0"
command = "python train.py config/train_wikitext.py --compile=False --wandb_log=False --eval_iters=10 --log_interval=1 --block_size=1024 --max_iters=500 --prune_type=reduce --prune_at=100 --lr_decay_iters=500 --batch_size=8 --n_layer=4 --n_head=4 --n_embd=128 --out_dir=pruned_models/wikitext_reduce_100_0.9_8/ --dropout=0.0"
           #python train.py config/train_shakespeare.py --device=cpu --compile=False --eval_iters=5 --log_interval=1 --block_size=64 --batch_size=1 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=5 --lr_decay_iters=5 --prune_type=reduce --prune_at=3 --dropout=0.0
result = subprocess.run(command,  shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
print("Done Training")
for prune_percentile in [0.9]:
    cur_model = "reduce" + "_" + str(prune_percentile)
    print("LOADING RESULTS ON ", cur_model)
    model_results = {}
    model_results[cur_model] = {}
    model_8_dir = "pruned_models/wikitext_reduce_100_0.9_8/ckpt.pt"
    #model_4_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_4/ckpt.pt"
    #model_12_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_12/ckpt.pt"

    model_8 = torch.load(model_8_dir, map_location='cpu')       #devi e
    #model_12 = torch.load(model_12_dir, map_location=device)
    #model_4 = torch.load(model_4_dir, map_location=device)

    print(" 8-batch", model_8)
    #print(" 4-batch", model_4)
    #print(" 12-batch", model_12)
    model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
    #model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
    #model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]
    model_results[cur_model]['training_throughput_8'] = model_8["iter_num"] * model_8["tokens_per_iter"] / model_8["total_time"]
    print(model_results)
    with open("run_all.json", "w") as outfile:
        json.dump(model_results, outfile)


# def experiment_train(params):
#     print(params['max_iters'])
#     prune_type = params['prune_type']
#     prune_at = params['prune_at']
#     prune_percentile = params['prune_percentile']
#     command = "python train.py" + " " + str(params['config_file']) + " --device=cpu --prune_type=\'" + str(prune_type) + "\' --prune_percentile=" + str(prune_percentile) + " --prune_at=" + str(params['prune_at']) + " --eval_iters=" + str(params['eval_iters']) + " --log_interval=" + str(params['log_interval']) + " --max_iters=" + str(params['max_iters']) + " --lr_decay_iters=" + str(params['lr_decay_iters']) + " --dropout=" + str(params['dropout']) + " --out_dir=\'" + str(params['out_dir']) + "\'"
#     print(command)
#     print("Running,", prune_type, prune_percentile, prune_at)
#     result = subprocess.run(command,  shell=True, capture_output=True, text=True)
#     print(result.stdout)
#     print(result.stderr)
#     print("Completed,", prune_type, prune_percentile, prune_at)

# # def experiment_test(params):
# #     command = f"python sample.py --out_dir={params['out_dir']} --start={params['start']} --num_samples={params['num_samples']} --max_new_tokens={params['max_new_tokens']} --device=cpu"
# #     time1 = time.time()
# #     result = subprocess.run(command, shell=True, capture_output=True, text=True)
# #     time2 = time.time()
# #     return result

# prune_types = ["reduce", "None", "zero_row", "zero_indiv"] # one instance: reduce and zero_row, another instance is None and zero_indiv
# prune_percentiles = [0.9, 0.5, 0.1]
# train_batches = [8]
# dataset = 'shakespeare'
# out_dir = f"pruned_models/{dataset}"
# cur_params = {"config_file": f"config/train_{dataset}.py", "eval_iters": 10, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 8, "prune_type": "None", "prune_at": 1, "prune_percentile": 0.9, "out_dir": out_dir} #max iters 500, prune at 100

# # All Models
# # prune_at = 100
# prune_at = 1
# model_dirs = {}
# model_results = {}
# for batch_size in train_batches:
#     cur_model_dirs = []
#     # Baseline Model
#     for prune_type in ["reduce"]: 

#         if prune_type == "None":
#             cur_out_dir = out_dir + "_" + prune_type + "_" + str(batch_size)
#             cur_params['out_dir'] = cur_out_dir
#             cur_params['batch_size'] = batch_size
#             cur_params['prune_type'] = "None"
#             #cur_params['max_iters'] = 500
#             cur_params['max_iters'] = 2
#             experiment_train(cur_params)
#         else:
#             print(';LADSJK;;LDFSF')
#             for prune_percentile in prune_percentiles:
#                 cur_out_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_" + str(batch_size)
#                 cur_params['out_dir'] = cur_out_dir
#                 cur_params['prune_type'] = prune_type
#                 cur_params['prune_at'] = prune_at
#                 cur_params['prune_percentile'] = prune_percentile
#                 cur_params['batch_size'] = batch_size
#                 cur_params['max_iters'] = 2
#                 experiment_train(cur_params)
#                 cur_model_dirs.append(cur_out_dir)
#         print("DONE TRAINING", prune_type)
#         if prune_type == "None":
#             cur_model = prune_type
#             print("LOADING RESULTS ON ", cur_model)
#             model_results[cur_model] = {}
#             model_8_dir = out_dir + "_" + prune_type + "_8/ckpt.pt"
#             # model_12_dir = out_dir + "_" + prune_type + "_12/ckpt.pt"
#             # model_4_dir = out_dir + "_" + prune_type + "_4/ckpt.pt"

#             model_8 = torch.load(model_8_dir, map_location='cpu')
#             # model_12 = torch.load(model_12_dir, map_location=device)
#             # model_4 = torch.load(model_4_dir, map_location=device)
#             print(" 8-batch", model_8)
#             # print(" 4-batch", model_4)
#             # print(" 12-batch", model_12)
#             model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
#             # model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
#             # model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]
#             model_results[cur_model]['training_throughput_8'] = model_8["iter_num"] * model_8["tokens_per_iter"] / model_8["total_time"]
#             print(model_results)
#             with open("run_all.json", "w") as outfile:
#                 json.dump(model_results, outfile)

        # else:
        #     for prune_percentile in prune_percentiles:
                
        #         cur_model = prune_type + "_" + str(prune_percentile)
        #         print("LOADING RESULTS ON ", cur_model)
        #         model_results[cur_model] = {}
        #         model_8_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_8/ckpt.pt"
        #         #model_4_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_4/ckpt.pt"
        #         #model_12_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_12/ckpt.pt"

        #         model_8 = torch.load(model_8_dir, map_location='cpu')       #devi e
        #         #model_12 = torch.load(model_12_dir, map_location=device)
        #         #model_4 = torch.load(model_4_dir, map_location=device)

        #         print(" 8-batch", model_8)
        #         #print(" 4-batch", model_4)
        #         #print(" 12-batch", model_12)
        #         model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
        #         #model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
        #         #model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]
        #         model_results[cur_model]['training_throughput_8'] = model_8["iter_num"] * model_8["tokens_per_iter"] / model_8["total_time"]
        #         print(model_results)
        #         with open("run_all.json", "w") as outfile:
        #             json.dump(model_results, outfile)
        # print("DONE EVALUATING", prune_type)
#     model_dirs[batch_size] = cur_model_dirs


# print("DONE WITH TRAINING and Eval")
# print(model_dirs)

# '''
# for prune_type in prune_types:
#     if prune_type == "None":
#         cur_model = prune_type
#         print("LOADING RESULTS ON ", cur_model)
#         model_results[cur_model] = {}
#         model_8_dir = out_dir + "_" + prune_type + "_8/ckpt.pt"
#         # model_12_dir = out_dir + "_" + prune_type + "_12/ckpt.pt"
#         # model_4_dir = out_dir + "_" + prune_type + "_4/ckpt.pt"

#         model_8 = torch.load(model_8_dir, map_location=device)
#         # model_12 = torch.load(model_12_dir, map_location=device)
#         # model_4 = torch.load(model_4_dir, map_location=device)
#         print(" 8-batch", model_8)
#         # print(" 4-batch", model_4)
#         # print(" 12-batch", model_12)
#         model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
#         # model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
#         # model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]
#         model_results[cur_model]['training_throughput_8'] = model_8["iter_num"] * model_8["tokens_per_iter"] / model_8["total_time"]

#     else:
#         for prune_percentile in prune_percentiles:
#             print("LOADING RESULTS ON ", cur_model)
#             cur_model = prune_type + "_" + prune_percentile
#             model_results[cur_model] = {}
#             model_8_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_8/ckpt.pt"
#             #model_4_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_4/ckpt.pt"
#             #model_12_dir = out_dir + "_" + prune_type + "_" + str(prune_at) + "_" + str(prune_percentile) + "_12/ckpt.pt"

#             model_8 = torch.load(model_8_dir, map_location=device)
#             #model_12 = torch.load(model_12_dir, map_location=device)
#             #model_4 = torch.load(model_4_dir, map_location=device)

#             print(" 8-batch", model_8)
#             #print(" 4-batch", model_4)
#             #print(" 12-batch", model_12)
#             model_results[cur_model]['val_loss'] = model_8["best_val_loss"]
#             #model_results[cur_model]['training_throughput_4'] = model_4["iter_num"] * model_4["tokens_per_iter"] / model_4["total_time"]
#             #model_results[cur_model]['training_throughput_12'] = model_12["iter_num"] * model_12["tokens_per_iter"] / model_12["total_time"]
#             model_results[cur_model]['training_throughput_8'] = model_8["iter_num"] * model_8["tokens_per_iter"] / model_8["total_time"]

# '''
# with open("run_all.json", "w") as outfile:
#     json.dump(model_results, outfile)