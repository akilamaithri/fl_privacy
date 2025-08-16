import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import os

# Get the JSON filename from command line argument or use default
if len(sys.argv) > 1:
    json_filename = sys.argv[1]
else:
    # Default filename - you can change this
    json_filename = "/scratch/wd04/sm0074/fl_privacy/logs/16aug/metrics/sst2_Exp.json"

# Read JSON data from file
try:
    with open(json_filename, 'r') as f:
        raw = f.read()
except FileNotFoundError:
    print(f"Error: File '{json_filename}' not found.")
    sys.exit(1)

# Parse JSON data
records = []
for line in raw.strip().splitlines():
    s = line.strip()
    if not s:
        continue
    if s.endswith(','):
        s = s[:-1]
    try:
        obj = json.loads(s)
        records.append(obj)
    except json.JSONDecodeError:
        continue

# Create DataFrame
df = pd.DataFrame.from_records(records)

# Function to create bar chart
def create_bar_chart(y_variable='noise_added'):
    """
    Create a bar chart with rounds on x-axis and specified variable on y-axis
    
    Parameters:
    y_variable: str, one of ['noise_added', 'epsilon', 'current_loss', 'noise_multiplier']
    """
    
    # Define colors for each client
    colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'}
    
    # Get unique rounds and clients
    rounds = sorted(df['round'].unique())
    clients = sorted(df['client_id'].unique())
    
    # Create a mapping from client_id to a consecutive index
    # client_to_index = {client_id: i for i, client_id in enumerate(clients)}
    n_clients = len(clients)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar width and positions
    bar_width = 0.2

    added_legend = set() # to only add each client once to the legend
    
    # Create bars for each client
    for round_num in rounds:
        rdata = df[df['round'] == round_num]
        present_clients = sorted(rdata['client_id'].unique())
        local_n = len(present_clients)  # number of bars in THIS round

        for j, client_id in enumerate(present_clients):
            cdata = rdata[rdata['client_id'] == client_id]
            if cdata.empty:
                continue
            val = cdata[y_variable].iloc[0]
            if y_variable in ("noise_added", "noise_multiplier"):
                val *= 128

            # offsets are computed with local_n so no gaps if someone is missing
            x = round_num + (j - local_n/2 + 0.5) * bar_width

            ax.bar(
                x, val, bar_width,
                label=(f'Client {client_id}' if client_id not in added_legend else None),
                color=colors.get(client_id, '#000000')
            )
            added_legend.add(client_id)
        
    # Customize the plot
    ax.set_xlabel('Round')
    ax.set_ylabel(y_variable.replace('_', ' ').title())
    ax.set_title(f'{y_variable.replace("_", " ").title()} by Round and Client (SST2)')
    ax.set_xticks(rounds)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

# Create the bar chart (you can change the y_variable parameter)
fig, ax = create_bar_chart(y_variable='epsilon')  # Change this to 'epsilon', 'current_loss', or 'noise_multiplier'

# Generate output filename based on input filename
base_name = os.path.splitext(os.path.basename(json_filename))[0]
output_filename = f"./tables/plot/client_comparison_{base_name}_.png"

# Save the plot
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.show()