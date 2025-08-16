import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Function to load data from a CSV file ---
# We use the pandas library, which is perfect for this.
# It reads the CSV into a table-like structure called a DataFrame.
def load_csv_file(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    # The 'strip' functions help clean up any accidental whitespace in the column names
    return pd.read_csv(file_path, skipinitialspace=True)

# --- 2. List your CSV files and desired labels ---
# Create a dictionary where the key is the label for the legend (e.g., "Strong Non-IID")
# and the value is the path to the CSV file.
# You can add up to 5 (or more) files here. Matplotlib will give each a different color.
files_to_plot = {
    "IID": "/scratch/wd04/sm0074/fl_privacy/no_noise/qnli_Iid_10_RDP.csv",
    "Linear": "/scratch/wd04/sm0074/fl_privacy/no_noise/qnli_Linear_10_RDP.csv",
    "Square": "/scratch/wd04/sm0074/fl_privacy/no_noise/sst2_Square_10_RDP.csv",
    # "Case 4: Balanced Non-IID": "path/to/your/case4_data.csv",
    # Add another file here if you have a fifth one
}

# --- 3. Create the plot ---
plt.figure(figsize=(12, 7)) # Adjust figure size for better readability

# Loop through each file in our dictionary
for label, path in files_to_plot.items():
    try:
        # Load the data from the CSV file
        data_df = load_csv_file(path)

        # --- 4. Extract the columns for plotting ---
        # We access columns by their names from the header row of the CSV.
        rounds = data_df['Round']
        
        # Convert accuracy from decimal to percentage
        accuracy = data_df['Accuracy'] * 100

        # Plot the data
        plt.plot(rounds, accuracy, label=label, marker='o', linestyle='-')

    except FileNotFoundError:
        print(f"Warning: Could not find the file at {path}. Skipping this plot.")
    except KeyError as e:
        print(f"Warning: Column {e} not found in {path}. Check your CSV headers. Skipping this plot.")


# --- 5. Customize and show the plot ---
plt.title("Federated Learning: Accuracy Comparison")
plt.xlabel("Rounds")
plt.ylabel("Accuracy (%)") # Updated label

legend_labels = ["Non DP", "Fixed Noise", "Dynamic Noise"]
plt.legend(legend_labels, loc="outside center", ncol=3, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.01))

# Set the y-axis limits to be between 70 and 100
plt.ylim(70, 100)

plt.grid(True)
# plt.legend() # This displays the labels for each line
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout

plt.savefig("tables/accuracy_plot.png")

plt.show()
