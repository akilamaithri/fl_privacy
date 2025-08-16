import json
import matplotlib.pyplot as plt

# Function to load JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load data from JSON files
data_FedMR_iid = load_json_file("../run/run-cifar10-fedmr-1/output/0/1_cifar10_vgg_250_FedMR_100_iid_1_accuracy_loss.json")
data_HClusteredSampling_iid = load_json_file("../run/run-cifar10-fedmr-1/output/0/1_cifar10_vgg_250_HClusteredSampling_100_iid_1_accuracy_loss.json")
data_HFedAvg_iid = load_json_file("../run/run-cifar10-fedmr-1/output/0/1_cifar10_vgg_250_HFedAvg_100_iid_1_accuracy_loss.json")
data_HFedProx_iid = load_json_file("../run/run-cifar10-fedmr-1/output/0/1_cifar10_vgg_250_HFedProx_100_iid_1_accuracy_loss.json")

data_FedMR_noniid_1 = load_json_file("../run/run-cifar10-fedmr-1/output/1/1_cifar10_vgg_250_FedMR_100_noniid_1_accuracy_loss.json")
data_HClusteredSampling_noniid_1 = load_json_file("../run/run-cifar10-fedmr-1/output/1/1_cifar10_vgg_250_HClusteredSampling_100_noniid_1_accuracy_loss.json")
data_HFedAvg_noniid_1 = load_json_file("../run/run-cifar10-fedmr-1/output/1/1_cifar10_vgg_250_HFedAvg_100_noniid_1_accuracy_loss.json")
data_HFedProx_noniid_1 = load_json_file("../run/run-cifar10-fedmr-1/output/1/1_cifar10_vgg_250_HFedProx_100_noniid_1_accuracy_loss.json")

data_FedMR_noniid_2 = load_json_file("../run/run-cifar10-fedmr-1/output/2/1_cifar10_vgg_250_FedMR_100_noniid_2_accuracy_loss.json")
data_HClusteredSampling_noniid_2 = load_json_file("../run/run-cifar10-fedmr-1/output/2/1_cifar10_vgg_250_HClusteredSampling_100_noniid_2_accuracy_loss.json")
data_HFedAvg_noniid_2 = load_json_file("../run/run-cifar10-fedmr-1/output/2/1_cifar10_vgg_250_HFedAvg_100_noniid_2_accuracy_loss.json")
data_HFedProx_noniid_2 = load_json_file("../run/run-cifar10-fedmr-1/output/2/1_cifar10_vgg_250_HFedProx_100_noniid_2_accuracy_loss.json")

data_FedMR_noniid_3 = load_json_file("../run/run-cifar10-fedmr-1/output/3/1_cifar10_vgg_250_FedMR_100_noniid_3_accuracy_loss.json")
data_HClusteredSampling_noniid_3 = load_json_file("../run/run-cifar10-fedmr-1/output/3/1_cifar10_vgg_250_HClusteredSampling_100_noniid_3_accuracy_loss.json")
data_HFedAvg_noniid_3 = load_json_file("../run/run-cifar10-fedmr-1/output/3/1_cifar10_vgg_250_HFedAvg_100_noniid_3_accuracy_loss.json")
data_HFedProx_noniid_3 = load_json_file("../run/run-cifar10-fedmr-1/output/3/1_cifar10_vgg_250_HFedProx_100_noniid_3_accuracy_loss.json")

data_FedMR_noniid_4 = load_json_file("../run/run-cifar10-fedmr-1/output/4/1_cifar10_vgg_250_FedMR_100_noniid_4_accuracy_loss.json")
data_HClusteredSampling_noniid_4 = load_json_file("../run/run-cifar10-fedmr-1/output/4/1_cifar10_vgg_250_HClusteredSampling_100_noniid_4_accuracy_loss.json")
data_HFedAvg_noniid_4 = load_json_file("../run/run-cifar10-fedmr-1/output/4/1_cifar10_vgg_250_HFedAvg_100_noniid_4_accuracy_loss.json")
data_HFedProx_noniid_4 = load_json_file("../run/run-cifar10-fedmr-1/output/4/1_cifar10_vgg_250_HFedProx_100_noniid_4_accuracy_loss.json")

# Extract accuracy data
rounds = [entry['round'] for entry in data_FedMR_iid['accuracy']]
data_FedMR_iid_accuracy = [entry['accuracy'] for entry in data_FedMR_iid['accuracy']]
data_HClusteredSampling_iid_accuracy = [entry['accuracy'] for entry in data_HClusteredSampling_iid['accuracy']]
data_HFedAvg_iid_accuracy = [entry['accuracy'] for entry in data_HFedAvg_iid['accuracy']]
data_HFedProx_iid_accuracy = [entry['accuracy'] for entry in data_HFedProx_iid['accuracy']]

data_FedMR_noniid_1_accuracy = [entry['accuracy'] for entry in data_FedMR_noniid_1['accuracy']]
data_HClusteredSampling_noniid_1_accuracy = [entry['accuracy'] for entry in data_HClusteredSampling_noniid_1['accuracy']]
data_HFedAvg_noniid_1_accuracy = [entry['accuracy'] for entry in data_HFedAvg_noniid_1['accuracy']]
data_HFedProx_noniid_1_accuracy = [entry['accuracy'] for entry in data_HFedProx_noniid_1['accuracy']]

data_FedMR_noniid_2_accuracy = [entry['accuracy'] for entry in data_FedMR_noniid_2['accuracy']]
data_HClusteredSampling_noniid_2_accuracy = [entry['accuracy'] for entry in data_HClusteredSampling_noniid_2['accuracy']]
data_HFedAvg_noniid_2_accuracy = [entry['accuracy'] for entry in data_HFedAvg_noniid_2['accuracy']]
data_HFedProx_noniid_2_accuracy = [entry['accuracy'] for entry in data_HFedProx_noniid_2['accuracy']]

data_FedMR_noniid_3_accuracy = [entry['accuracy'] for entry in data_FedMR_noniid_3['accuracy']]
data_HClusteredSampling_noniid_3_accuracy = [entry['accuracy'] for entry in data_HClusteredSampling_noniid_3['accuracy']]
data_HFedAvg_noniid_3_accuracy = [entry['accuracy'] for entry in data_HFedAvg_noniid_3['accuracy']]
data_HFedProx_noniid_3_accuracy = [entry['accuracy'] for entry in data_HFedProx_noniid_3['accuracy']]

data_FedMR_noniid_4_accuracy = [entry['accuracy'] for entry in data_FedMR_noniid_4['accuracy']]
data_HClusteredSampling_noniid_4_accuracy = [entry['accuracy'] for entry in data_HClusteredSampling_noniid_4['accuracy']]
data_HFedAvg_noniid_4_accuracy = [entry['accuracy'] for entry in data_HFedAvg_noniid_4['accuracy']]
data_HFedProx_noniid_4_accuracy = [entry['accuracy'] for entry in data_HFedProx_noniid_4['accuracy']]

# Create subplots
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
# fig.suptitle("Federated Learning: Accuracy Comparisons", fontsize=16)

# Define cases
cases = [
    (data_FedMR_iid_accuracy, data_HClusteredSampling_iid_accuracy, data_HFedAvg_iid_accuracy, data_HFedProx_iid_accuracy, "Case 0: IID"),
    (data_FedMR_noniid_1_accuracy, data_HClusteredSampling_noniid_1_accuracy, data_HFedAvg_noniid_1_accuracy, data_HFedProx_noniid_1_accuracy, "Case 1: Strong Non-IID"),
    (data_FedMR_noniid_2_accuracy, data_HClusteredSampling_noniid_2_accuracy, data_HFedAvg_noniid_2_accuracy, data_HFedProx_noniid_2_accuracy, "Case 2: Moderate Non-IID"),
    (data_FedMR_noniid_3_accuracy, data_HClusteredSampling_noniid_3_accuracy, data_HFedAvg_noniid_3_accuracy, data_HFedProx_noniid_3_accuracy, "Case 3: Mild Non-IID"),
    (data_FedMR_noniid_4_accuracy, data_HClusteredSampling_noniid_4_accuracy, data_HFedAvg_noniid_4_accuracy, data_HFedProx_noniid_4_accuracy, "Case 4: Balanced Non-II")
]

legend_labels = ["FedMR", "ClusteredSampling", "FedAvg", "FedProx"]

# Plot data for each case
for i, (fedmr, clustered, fedavg, fedprox, title) in enumerate(cases):
    ax = axes[i]  # Access the correct subplot
    ax.plot(rounds, fedmr, label="FedMR", marker='o')
    ax.plot(rounds, clustered, label="ClusteredSampling", marker='x')
    ax.plot(rounds, fedavg, label="FedAvg", marker='s')
    ax.plot(rounds, fedprox, label="FedProx", marker='d')
    ax.axvline(x=100, color='red', linestyle='--', label='Round 100')
    ax.text(102, max(fedmr) * 0.5, 'Model Recombination\nHas Started', color='red', fontsize=10, rotation=0, va='center')
    ax.set_title(title)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True)

fig.legend(legend_labels, loc="outside center", ncol=4, fontsize=12, frameon=False, bbox_to_anchor=(0.5, 0.01))


plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout
# fig.tight_layout(pad=4)
plt.show()

plt.savefig("tables/accuracy_plot.png")
