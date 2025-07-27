

# An Interactive Framework for Implementing Privacy-Preserving Federated Learning: Experiments on Large Language Models

## Notes

This project implements federated fine‑tuning of transformer models on GLUE datasets using Flower. It incorporates differential privacy techniques and contains utilities for computing noise levels and plotting results.

* The code adapts HuggingFace’s `run_glue` example and runs in the Flower federated learning framework.

* Because Flower 1.12.0 has an issue with local DP, the repository ships modified modules that must replace Flower’s defaults: `fixed/localdp_fixed_mod.py` and `fixed/differential_privacy.py`.

### Project Structure
|Folder|Content|
|--|--|
|conf/                 |Patched Flower modules used for local DP|
|fixed/                |Patched Flower modules used for local DP|
|noise_calculation/    |Scripts to compute sigma values for target epsilon|
|plot/.                |Scripts for plotting noise/accuracy curves|
|privacy_tools/        |Custom privacy accounting utilities|
|federated.py          |Main entry point for running simulations|
|utils.py              |Helper classes and Flower client utilities|
|script.sh             |Example bash script to launch an experiment|

Configuration for Flower resides in `conf/federated.yaml`, specifying number of clients, rounds, resources and DP clipping norms

### Differential Privacy
* `fixed/localdp_fixed_mod.py` defines a Flower “mod” that clips client updates and injects Gaussian noise before sending them to the server. 
* The noise depends on the client partition id and a predefined noise list
* Precomputed noise levels for different datasets and partition policies are in noise_epsilon_6.json and noise_epsilon_10.json.

### Running Experiments
* `federated.py` uses HuggingFace Transformers and Flower to run the simulation.
* It parses command-line arguments for model paths, dataset name, partition policy, DP accountant type, etc., and sets up the dataset partitions using `flwr_datasets`, then starts the server and clients.
* The code loads the per-partition noise values (e.g., from noise_epsilon_6.json) and passes them to LocalDpFixedMod for local DP.

### Noise Calculation
* `noise_calculation/get_noise.py` computes the sigma required to meet a target epsilon under different accounting modes (“wang”, “our”, “rdp”).
* It loops over candidate sigma values, evaluates the privacy bound, and logs the chosen noise.

### Utilities and Data Classes
* `utils.py` contains data classes for command‑line arguments, helpers to load models and datasets, and a HuggingFaceClient that implements the Flower client logic.
* It also includes a `get_config` function that loads Hydra configuration files.

<hr>

# Original ReadMe Content

The GLUE dataset learning process is using Transformers library and is adopted from [run_glue](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py). <br>
The Federated learning enviroment is using [Flower](https://flowerai.net/docs/framework/index.html). <br>


## NOTE
Because of an existing problem with flwr[simulation]==1.12.0 when using local DP, following steps should be done:

- Copy [localdp_fixed_mod.py](fixed/localdp_fixed_mod.py) in flwr/client/mod

- Use [differential_privacy.py](fixed/differential_privacy.py) instead of flwr/common/differential_privacy.py

## Install dependencies
```
pip install -r requirement.txt
```
## Experiments
To run the experiments in the paper run:
```
./script.sh
```
## Noise Calculation
To calculate required noise for target Epsilon, we used our previous work [FSRDP Accountant](https://github.com/star-ailab/FSRDP).
```
Python ./noise_calculation/get_noise.py
```
target_epsilons and dataset_size_list is configurable in get_noise.py file.

## Single Experiment
```
python federated.py \
  --model_name_or_path google-bert/bert-base-cased \
  --max_seq_length 128 \
  --task_name SST2 \
  --partition_policy Linear \
  --per_device_train_batch_size 550 \
  --learning_rate 2e-5\
  --output_dir /tmp/SST2/
```
- Model_name is the based model. <br>
- task_name is the dataset which can be (SST2, QNLI, or QQP).<br>
- Parition_policy can be (Iid, Linear, Square, or Exp).
## citation
Please cite our papar if you find our repo helpful.

```
@misc{ahmadi2025interactiveframeworkimplementingprivacypreserving,
      title={An Interactive Framework for Implementing Privacy-Preserving Federated Learning: Experiments on Large Language Models}, 
      author={Kasra Ahmadi and Rouzbeh Behnia and Reza Ebrahimi and Mehran Mozaffari Kermani and Jeremiah Birrell and Jason Pacheco and Attila A Yavuz},
      year={2025},
      eprint={2502.08008},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.08008}, 
}
```
