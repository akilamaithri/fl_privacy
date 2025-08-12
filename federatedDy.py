# federated.py (patched for dynamic noise broadcast)

import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import csv
import random
import numpy as np
import evaluate
import datasets
import flwr as fl

from logging import WARNING, ERROR, LogRecord
from datasets import disable_caching, enable_caching
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner, LinearPartitioner, SquarePartitioner, ExponentialPartitioner, SizePartitioner
)

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

from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# --- our modules ---
from utils import *  # brings ModelArguments, DataTrainingArguments, DpArguments, get_config, etc.
from dynamic_strategy import DynamicNoiseStrategy  # NEW

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

# --- Parse args (add DpArguments) ---
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, DpArguments))
model_args, data_args, training_args, dp_args = parser.parse_args_into_dataclasses()
cfg = get_config("federated")

# -------------------- Data prep (unchanged) --------------------
raw_datasets = load_dataset(
    "nyu-mll/glue",
    data_args.task_name,
    cache_dir="/scratch/wd04/sm0074/hf_cache",
    token=model_args.token,
)

is_regression = data_args.task_name == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
else:
    num_labels = 1

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
    from_tf=bool(".ckpt" in model_args.model_name_or_path),
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    token=model_args.token,
    trust_remote_code=model_args.trust_remote_code,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    local_files_only=True,
)
initial_param = [val.cpu().numpy() for _, val in model.state_dict().items()]
print(type(initial_param))

sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
padding = "max_length" if data_args.pad_to_max_length else False

label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and data_args.task_name is not None
    and not is_regression
):
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if sorted(label_name_to_id.keys()) == sorted(label_list):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warning(
            "Model labels don't match dataset labels; ignoring model labels."
        )
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
        f"max_seq_length {data_args.max_seq_length} > model_max_length {tokenizer.model_max_length}; "
        f"using {tokenizer.model_max_length}."
    )
max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

with training_args.main_process_first(desc="dataset map pre-processing"):
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]

if data_args.pad_to_max_length:
    data_collator = default_data_collator
elif training_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None

metric = evaluate.load("glue", data_args.task_name)

# -------------------- CSV output (unchanged) --------------------
file_name = f"{data_args.task_name}_{data_args.partition_policy}_{data_args.epsilon}_{data_args.accountant}.csv"
model_performance_file = "./logs/dyNoise_CSV/" + file_name
headers = ['Round', 'Accuracy', 'Info']
with open(model_performance_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

# -------------------- Server evaluate_fn (unchanged) --------------------
def server_fn(context: Context):
    # Use our DynamicNoiseStrategy and feed scheduler bounds from dp_args
    strategy = DynamicNoiseStrategy(
        # scheduler knobs (public, no privacy cost)
        loss_scale_lo=dp_args.loss_scale_lo,
        loss_scale_hi=dp_args.loss_scale_hi,
        loss_scale_alpha=dp_args.loss_scale_k,   # reuse 'k' as alpha
        # FedAvg kwargs (same as before)
        min_available_clients=cfg.flower.num_clients,
        fraction_fit=cfg.flower.fraction_fit,
        fraction_evaluate=0,
        initial_parameters=ndarrays_to_parameters(initial_param),
        evaluate_fn=get_evaluate_fn(
            model_args,
            1,
            cfg.flower.num_rounds,
            "./result_model",
            training_args,
            eval_dataset,
            tokenizer,
            data_collator,
            metric,
            model_performance_file,
        ),
    )

    num_rounds = cfg.flower.num_rounds
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

# -------------------- FederatedDataset partitions (unchanged) --------------------
if (data_args.partition_policy == "Iid"):
    partitioner = IidPartitioner(num_partitions=cfg.flower.num_clients)
elif (data_args.partition_policy == "Linear"):
    partitioner = LinearPartitioner(num_partitions=cfg.flower.num_clients)
elif (data_args.partition_policy == "Square"):
    partitioner = SquarePartitioner(num_partitions=cfg.flower.num_clients)
elif (data_args.partition_policy == "Exp"):
    partitioner = ExponentialPartitioner(num_partitions=cfg.flower.num_clients)
elif (data_args.partition_policy == "Manual"):
    partition_sizes = [1000, 1000, 1000, 1000]
    partitioner = SizePartitioner(partition_sizes)
else:
    raise ValueError(f"Unknown partition policy: {data_args.partition_policy}")

fds = FederatedDataset(
    dataset="nyu-mll/glue",
    subset=data_args.task_name,
    partitioners={"train": partitioner}
)
label_list = fds.load_partition(0).features["label"].names

# -------------------- DP mod (temporarily disabled) --------------------
# We will enable a per-round DP mod in the next step.
# The old per-client fixed mod is kept here for reference:
# from fixed.localdp_fixed_mod import LocalDpFixedMod
# with open('noise_epsilon_10.json', 'r') as file:
#     noise = json.load(file)
# noise_list = noise[data_args.task_name][data_args.partition_policy][data_args.accountant]
# if len(noise_list) != cfg.flower.num_clients:
#     exit()
# local_dp_obj = LocalDpFixedMod(cfg.flower.dp.clipping_norm, noise_list=noise_list)

# -------------------- Build client & server apps --------------------
client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        training_args,
        model_args,
        data_args,
        label_list,
        dp_args=dp_args,   # pass through so clients can compute sigma_t
    ),
    # mods=[local_dp_obj],  # will enable a dynamic mod in the next patch
)

server = fl.server.ServerApp(server_fn=server_fn)

client_resources = dict(cfg.flower.client_resources)
fl.simulation.run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=cfg.flower.num_clients,
    backend_config={"client_resources": client_resources},
)