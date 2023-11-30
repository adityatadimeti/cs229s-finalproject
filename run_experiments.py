 # run the training script with the basic configs from the file
# script_path = 'train.py'
# config_path = 'config/train_wikitext.py'
# out_dir = "--out_dir=" + config_dict["out_dir"]
# device = "--device=" + str(config_dict["device"])
# max_iters = "--max_iters=" + str(500)
# lr_decay_iters = "--lr_decay_iters=" + str(500)
# block_size = "--block_size=" + str(config_dict["block_size"])
# batch_size = "--batch_size=" + str(config_dict["batch_size"])
#
# # Construct the command string
# command = f'python {script_path} {config_path} {out_dir} {device} {max_iters} {lr_decay_iters} {block_size} {batch_size}'
# # Execute the command
# subprocess.run(command, shell=True)
import subprocess
import ast
import json
import time


def experiment_train(params):
    command = f"python train.py {params['config_file']} --eval_iters={params['eval_iters']} --log_interval={params['log_interval']}  --max_iters={params['max_iters']} --lr_decay_iters={params['lr_decay_iters']} --dropout={params['dropout']}"
    print(command)
    command = f"python train.py config/train_shakespeare.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2 --lr_decay_iters=2000 --dropout=0.0"
    time1 = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    time2 = time.time()
    print(result.stdout)
    return result, (time2 - time1)

def experiment_test(params):
    command = f"python sample.py --out_dir={params['out_dir']} --start={params['start']} --num_samples={params['num_samples']} --max_new_tokens={params['max_new_tokens']} --device=cpu"
    time1 = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    time2 = time.time()
    return result

#train_params_8 = {"config_file": "config/train_wikitext.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 8}
#train_params_1 = {"config_file": "config/train_wikitext.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 1}
#train_params_12 = {"config_file": "config/train_wikitext.py", "eval_iters": 100, "log_interval": 1, "max_iters": 2, "lr_decay_iters": 500, "dropout": 0.0, "batch_size": 12}
#train_result_8, train_time_8 = experiment_train(train_params_8)

test_params_1 = {"out_dir": "shakespeare", "start": "\'What is the answer to life, the universe, and everything?\'", "num_samples": 1, "max_new_tokens": 100}
test_params_12 = {"out_dir": "shakespeare", "start": "\'What is the answer to life, the universe, and everything?\'", "num_samples": 12, "max_new_tokens": 100}
test_result_1 = experiment_test(test_params_1)
test_result_12 = experiment_test(test_params_12)

#train_8_logs = str(train_result_8.stdout).split(" ")
#last_occurrence_index = len(train_8_logs) - train_8_logs[::-1].index("val") - 1
#val_loss = float(train_8_logs[last_occurrence_index+2])
# train_samples
#train_throughput = train_samples / train_time
print(test_result_1.stdout)
inference_latency_1 = float(str(test_result_1.stdout).split()[-2])
inference_latency_12 = float(str(test_result_12.stdout).split()[-2])
print(inference_latency_1, inference_latency_12)
metric_dict = {"inference_latency_1": inference_latency_1, "inference_latency_12": inference_latency_12}
with open("run.json", "w") as outfile:
    json.dump(metric_dict, outfile)
