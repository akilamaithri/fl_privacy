import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import flwr as fl
import evaluate
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner,LinearPartitioner,SquarePartitioner,ExponentialPartitioner,SizePartitioner
from datasets import load_dataset
from flwr.client.mod import fixedclipping_mod
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping
# LocalDpMod is used to add local differential privacy to the client side.
from flwr.client.mod.localdp_mod import LocalDpMod    
# LocalDpFixedMod is used to add local differential privacy to the client side with fixed clipping and noise.
from fixed.localdp_fixed_mod import LocalDpFixedMod
from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import os
import random
import sys
from logging import WARNING, ERROR, LogRecord
import datasets
import numpy as np
from datasets import disable_caching, enable_caching
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from utils import *
import csv

os.environ["HF_DATASETS_OFFLINE"] = "1"

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
# Setup logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
)

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses() 
cfg = get_config("federated") #loads from conf/federated.yaml

#Server config
raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir="/scratch/wd04/sm0074/hf_cache",
            token=model_args.token,
        )

#checks if task is regression (stsb) or classification (all others)
is_regression = data_args.task_name == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
else:
    num_labels = 1
#end_preapare_data

# Load pretrained model and tokenizer
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir, 
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code, 
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer, 
    revision=model_args.model_revision, 
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    from_tf=bool(".ckpt" in model_args.model_name_or_path), # checks if the model is a TensorFlow model
    config=config, # loads the config from the model_args
    cache_dir=model_args.cache_dir, 
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    local_files_only=True, 
)
# intial global model parameters
# This is used to initialize the global model on the server side
# and to compare the performance of the model at each round.
# It is also used to check if the model has been updated after each round.
initial_param = [val.cpu().numpy() for _, val in model.state_dict().items()]
print(type(initial_param))

#Pre-processing
############################################################################################################################
# Preprocessing the raw_datasets
sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
# Padding strategy
if data_args.pad_to_max_length:
    padding = "max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False
# Some models have set the order of the labels to use, so let's make sure we do use it.
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and data_args.task_name is not None
    and not is_regression
):
    # This is to ensure that the label names in the config match the label names in the dataset
    # This is important for the model to be able to map the labels correctly
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    # If the label names in the config match the label names in the dataset, we can use them
    # Otherwise, we will ignore the model labels and use the dataset labels
    # This is to ensure that the model can be used with the dataset labels
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: "
            f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
            "\nIgnoring the model labels as a result.",
        )
# elif data_args.task_name is not None and is_regression:
#     # For regression tasks, we don't need to set label_to_id
elif data_args.task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}
if label_to_id is not None:
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in config.label2id.items()}
elif data_args.task_name is not None and not is_regression:
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {id: label for label, id in config.label2id.items()}
if data_args.max_seq_length > tokenizer.model_max_length:
    logger.warning(
        f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
# Set the max_seq_length to the minimum of the data_args.max_seq_length and the tokenizer.model_max_length
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
# Tokenize the dataset and prepare it for training
def preprocess_function(examples):
    # Tokenize the texts
    args = (
    (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result
# Uused to ensure that the dataset is processed only once when using distributed training. 
# It prevents multiple processes from trying to process the dataset at the same
# time, which can lead to data corruption or other issues.
with training_args.main_process_first(desc="dataset map pre-processing"):
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
# Used to get the validation dataset for evaluation.
# If the task is MNLI, we use the "validation_matched" split, otherwise we use the "validation" split.
# This is because MNLI has two validation sets: "validation_matched" and "validation_mismatched".
# The "validation_matched" set is used for the matched validation set, while the "validation" set is used for the mismatched validation set.
# The eval_dataset is used for evaluation during training and for final evaluation after training.
# It is used to compute the metrics for the model, such as accuracy, F1 score, etc.
# The eval_dataset is also used to compute the loss for the model during training.
eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if data_args.pad_to_max_length:
    data_collator = default_data_collator
elif training_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None
metric = evaluate.load("glue", data_args.task_name)

#ready to store the results
file_name = data_args.task_name+"_"+data_args.partition_policy+"_"+str(data_args.epsilon)+"_"+data_args.accountant+".csv"
# file_name = data_args.task_name+"_"+data_args.partition_policy+"_no_noise"+".csv"

# model_performance_file = "./performance/DP_local_fixed/"+file_name
model_performance_file = "./"+file_name
headers = ['Round', 'Accuracy','Info']

# Open the CSV file in write mode and add headers (this will overwrite if the file already exists)
with open(model_performance_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Writing headers

def server_fn(context: Context):
    # Define the Strategy
    strategy = fl.server.strategy.FedAvg(
    min_available_clients=cfg.flower.num_clients, # total clients
    fraction_fit=cfg.flower.fraction_fit, # ratio of clients to sample
    fraction_evaluate=0, # No federated evaluation
    initial_parameters=ndarrays_to_parameters(initial_param),
        # A (optional) function used to configure a "fit()" round
    # on_fit_config_fn=get_on_fit_config(),
        # A (optional) function to aggregate metrics sent by clients
    # fit_metrics_aggregation_fn=fit_weighted_average,
        # A (optional) function to execute on the server after each round. 
        # In this example the function only saves the global model.
    evaluate_fn=get_evaluate_fn( 
            model_args,
            1,
            cfg.flower.num_rounds,
            "./result_model",
            training_args, #checktraining_args
            eval_dataset,
            tokenizer,
            data_collator,
            metric,
            model_performance_file
    ),
    )

    # # Add Differential Privacy
    # sampled_clients = cfg.flower.num_clients*strategy.fraction_fit
    # strategy = DifferentialPrivacyClientSideFixedClipping(
    #     strategy, 
    #     noise_multiplier=cfg.flower.dp.noise_mult,
    #     clipping_norm=cfg.flower.dp.clip_norm, 
    #     num_sampled_clients=sampled_clients
    # )
    #Local DP


    # Number of rounds to run the simulation
    num_rounds = cfg.flower.num_rounds
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config) 

#Dataset
if(data_args.partition_policy == "Iid"):
    partitioner = IidPartitioner(num_partitions=cfg.flower.num_clients)
elif(data_args.partition_policy == "Linear"):
    partitioner = LinearPartitioner(num_partitions=cfg.flower.num_clients)
elif(data_args.partition_policy == "Square"):
    partitioner = SquarePartitioner(num_partitions=cfg.flower.num_clients)
elif(data_args.partition_policy == "Exp"):
    partitioner = ExponentialPartitioner(num_partitions=cfg.flower.num_clients)
elif(data_args.partition_policy == "Manual"):
    partition_sizes = [1000,1000,1000,1000]
    partitioner = SizePartitioner(partition_sizes)
    
# was "nyu-mll/glue"
fds = FederatedDataset(
    dataset="nyu-mll/glue",
    subset=data_args.task_name,
    partitioners={"train": partitioner}
)
label_list = fds.load_partition(0).features["label"].names
num_labels = len(label_list)
# visualize_partitions(fds_train)
# local_dp_obj = LocalDpMod(cfg.flower.dp.clipping_norm, 1.5, cfg.flower.dp.epsilon, cfg.flower.dp.delta)

# dataset was suggested to be commented by codex
dataset_size = len(fds.load_partition(0))   

my_sum = 0
for index in range(cfg.flower.num_clients):
    my_sum += len(fds.load_partition(index))
    print("Partition {}: {}".format(index,len(fds.load_partition(index))))
print("Overal is {}".format(my_sum))
noise_list = []
for index in range(cfg.flower.num_clients):
    batch_size = training_args.per_device_train_batch_size
    dataset_size = len(fds.load_partition(index))
    q = batch_size/dataset_size
    steps = cfg.flower.num_rounds * (dataset_size//batch_size)
    delta = 1/dataset_size
    sigma, eps = get_sigma(q, steps, data_args.epsilon, delta,init_sigma=2, interval=0.5, mode=data_args.accountant)
    print('noise std:', sigma, 'eps: ', eps)
    noise_list.append(sigma)
exit()

local_dp_obj = LocalDpMod(cfg.flower.dp.clipping_norm, 1.5, data_args.epsilon, delta)

if data_args.accountant.lower() == "rdp":
    dataset_size = len(fds.load_partition(0))
    delta = 1 / dataset_size
    local_dp_obj = LocalDpMod(
        clipping_norm=cfg.flower.dp.clipping_norm,
        sensitivity=cfg.flower.dp.sensitivity,
        epsilon=data_args.epsilon,
        delta=delta
    )
else:
    with open('noise_epsilon_6.json', 'r') as file:
        noise = json.load(file)
    noise_list = noise[data_args.task_name][data_args.partition_policy][data_args.accountant]
    local_dp_obj = LocalDpFixedMod(cfg.flower.dp.clipping_norm, noise_list)
    
# Build list of noises from JSON
# with open('noise_epsilon_10.json', 'r') as file:
#     noise = json.load(file)
# noise_list = noise[data_args.task_name][data_args.partition_policy][data_args.accountant]

# epsilon_list = [data_args.epsilon] * cfg.flower.num_clients
# delta_list = [1.0 / dataset_size] * cfg.flower.num_clients

# if(len(noise_list)!=4):
#     exit()

# if len(noise_list) != cfg.flower.num_clients:
#     exit()

# local_dp_obj = LocalDpFixedMod(cfg.flower.dp.clipping_norm, epsilon_list, delta_list) 

# local_dp_obj = LocalDpFixedMod(cfg.flower.dp.clipping_norm, noise_list)   

# codex suggestion - Load noise values and and constrcut LocalDpFixedMod
# This is used to initialize the local differential privacy object with the clipping norm and noise list.
# The clipping norm is used to clip the gradients, and the noise list is used to add noise to the gradients.
# The noise is used to ensure that the gradients are differentially private.
# local_dp_obj = LocalDpFixedMod(
#     cfg.flower.dp.clipping_norm,
#     noise_list=noise_list,
# )

# This is used to initialize the client function with the federated dataset, training arguments, 
# model arguments, data arguments, and label list.
client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        training_args,
        model_args,
        data_args,
        label_list
    ),
    mods=[local_dp_obj] 
)

# This is used to initialize the server function with the model, training arguments, and data arguments.
# The server function is used to run the federated learning simulation.
server = fl.server.ServerApp(server_fn=server_fn)
# This is used to run the federated learning simulation with the server and client functions.
# It runs the simulation for the specified number of rounds and uses the specified client resources.
# The client resources are used to specify the resources available to each client, such as CPU and memory.
# The backend_config is used to specify the backend configuration for the simulation, such as the
client_resources = dict(cfg.flower.client_resources)
fl.simulation.run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=cfg.flower.num_clients,
    backend_config={"client_resources": client_resources},

)
