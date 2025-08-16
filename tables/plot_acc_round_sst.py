import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Updated function to load data from a CSV file ---
# This function reads your CSV files.
def load_csv_data(file_path):
    """
    Loads data from a specified CSV file.
    Returns the 'Round' and 'Accuracy' columns.
    """
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        rounds = df['Round']
        accuracy = df['Accuracy'] * 100 # Convert to percentage
        return rounds, accuracy
    except FileNotFoundError:
        print(f"Warning: File not found at {file_path}. Plot will be empty.")
        return pd.Series(dtype='int'), pd.Series(dtype='float') # Return empty pandas Series
    except KeyError as e:
        print(f"Warning: Column {e} not found in {file_path}. Check your CSV headers.")
        return pd.Series(dtype='int'), pd.Series(dtype='float')


# --- 2. Define the structure of your plots and data files ---
# This list defines what to draw in each of the four subplots.
plot_configs = [
    {
        "title": "IID",
        "files": {
            "No Noise": "/scratch/wd04/sm0074/fl_privacy/logs/15aug/csv/non/sst2_Iid.csv",
            "Fixed Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csvFx/sst2_Iid.csv",
            "Dynamic Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csv/sst2_Iid.csv",
        }
    },
    {
        "title": "Linear",
        "files": {
            "No Noise": "/scratch/wd04/sm0074/fl_privacy/logs/15aug/csv/non/sst2_Linear.csv",
            "Fixed Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csvFx/sst2_Linear_ep17.csv",
            "Dynamic Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csv/sst2_Linear.csv",
        }
    },
    {
        "title": "Square",
        "files": {
            "No Noise": "/scratch/wd04/sm0074/fl_privacy/logs/15aug/csv/non/sst2_Square.csv",
            "Fixed Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csvFx/sst2_Square_ep17.csv",
            "Dynamic Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csv/sst2_Square2.csv",
        }
    },
    {
        "title": "Exp",
        "files": {
            "No Noise": "/scratch/wd04/sm0074/fl_privacy/logs/15aug/csv/non/sst2_Exp.csv",
            "Fixed Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csvFx/sst2_Exp_ep17.csv",
            "Dynamic Noise": "/scratch/wd04/sm0074/fl_privacy/logs/16aug/csv/sst2_Exp3.csv",
        }
    }
]

# --- 3. Create subplots with original styling ---
# We create 1 row and 4 columns for our plots, using the original figsize.
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# --- 4. Plot data for each case ---
# This loop goes through each of the four plot configurations.
for i, config in enumerate(plot_configs):
    ax = axes[i]  # Get the correct subplot (ax) for the current case

    # Load the data for the three lines
    rounds, no_noise_acc = load_csv_data(config["files"]["No Noise"])
    _, fixed_noise_acc = load_csv_data(config["files"]["Fixed Noise"])
    _, dynamic_noise_acc = load_csv_data(config["files"]["Dynamic Noise"])

    # Plot the three lines on the subplot if data was loaded successfully
    if not rounds.empty:
        ax.plot(rounds, no_noise_acc, label="No Noise", marker='o')
        ax.plot(rounds, fixed_noise_acc, label="Fixed Noise", marker='x', color='red')
        ax.plot(rounds, dynamic_noise_acc, label="Dynamic Noise", marker='*', color='green')

    # --- 5. Customize each subplot ---
    ax.set_title(config["title"])
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 100)
    ax.set_xlim(left=0)
    ax.grid(True)


# --- 6. Add a single, shared legend with original styling ---
legend_labels = ["No Noise", "Fixed Noise", "Dynamic Noise"]
# legend_labels = ["No Noise", "Dynamic Noise"]
# Using the exact legend placement from your first script
fig.legend(legend_labels, loc="outside center", ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.01))

# Adjust layout with original parameters
plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- 7. Save and show the final plot ---
plt.savefig("./tables/plot/accuracy_round_sst_.png")
plt.show()
