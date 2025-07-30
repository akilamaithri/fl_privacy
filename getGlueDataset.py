import os

# Set Hugging Face cache locations to /scratch BEFORE importing anything
scratch_cache = "/scratch/wd04/sm0074/hf_cache"
os.environ["HF_HOME"] = scratch_cache
os.environ["HF_DATASETS_CACHE"] = scratch_cache
os.environ["TRANSFORMERS_CACHE"] = scratch_cache
os.environ["HF_HUB_CACHE"] = scratch_cache
os.environ["HF_HUB_OFFLINE"] = "0"  # or "0" if you're online

from datasets import load_dataset

# Try loading QQP dataset from cache
ds = load_dataset("glue", "qnli", cache_dir=scratch_cache)

print(ds)
