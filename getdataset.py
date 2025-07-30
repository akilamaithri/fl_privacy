import os

# Set the cache directories
scratch_cache = "/scratch/wd04/sm0074/hf_cache"
os.environ["HF_HOME"] = scratch_cache
os.environ["HF_DATASETS_CACHE"] = scratch_cache
os.environ["TRANSFORMERS_CACHE"] = scratch_cache
os.environ["HF_HUB_CACHE"] = scratch_cache
os.environ["MPLCONFIGDIR"] = os.path.join(scratch_cache, "mpl_cache")
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

from datasets import load_dataset
import evaluate # Import evaluate

print(f"Downloading to HF_HOME: {os.environ.get('HF_HOME')}")

print("Downloading nyu-mll/glue")
try:
    # --- CHANGE THIS LINE ---
    load_dataset("nyu-mll/glue", "qnli") # <-- Make sure you're asking for 'sst2' here
    # -----------------------
    print("nyu-mll/glue with config 'qnli' downloaded and cached successfully.")
except Exception as e:
    print(f"Error during download: {e}")

print("\nDownloading 'glue' evaluation metric...")
try:
    metric = evaluate.load("glue", config_name="qnli") # Specify config_name if needed, often not for metric itself
    print("'glue' evaluation metric downloaded and cached successfully.")
except Exception as e:
    print(f"Error during metric download: {e}")