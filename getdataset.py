import os
from datasets import load_dataset
import evaluate # Import evaluate

# Set the cache directories
os.environ["HF_HOME"] = "/scratch/wd04/sm0074/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/wd04/sm0074/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/wd04/sm0074/hf_cache"
os.environ["MPLCONFIGDIR"] = "/scratch/wd04/sm0074/hf_cache/mpl_cache"
os.environ["HF_DATASETS_OFFLINE"] = "0" # Ensure it's not offline for download

print(f"Downloading to HF_HOME: {os.environ.get('HF_HOME')}")

# print("Downloading nyu-mll/glue (this might take a while)...")
# try:
#     # --- CHANGE THIS LINE ---
#     load_dataset("nyu-mll/glue", "sst2") # <-- Make sure you're asking for 'sst2' here
#     # -----------------------
#     print("nyu-mll/glue with config 'sst2' downloaded and cached successfully.")
# except Exception as e:
#     print(f"Error during download: {e}")

print("\nDownloading 'glue' evaluation metric...")
try:
    metric = evaluate.load("glue", config_name="sst2") # Specify config_name if needed, often not for metric itself
    print("'glue' evaluation metric downloaded and cached successfully.")
except Exception as e:
    print(f"Error during metric download: {e}")