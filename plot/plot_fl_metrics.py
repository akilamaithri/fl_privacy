import pandas as pd
import matplotlib.pyplot as plt
import json

# Replace this with your CSV content or read from a file
csv_data = """Round,Accuracy,Info
5,0.5091743119266054,"{'eval_loss': 97.30390167236328, 'eval_accuracy': 0.5091743119266054, 'eval_runtime': 3.178, 'eval_samples_per_second': 274.384, 'eval_steps_per_second': 17.306}"
1,0.5091743119266054,"{'eval_loss': 43.40767288208008, 'eval_model_preparation_time': 0.0032, 'eval_accuracy': 0.5091743119266054, 'eval_runtime': 3.0564, 'eval_samples_per_second': 285.308, 'eval_steps_per_second': 35.663}"
6,0.5091743119266054,"{'eval_loss': 133.29977416992188, 'eval_accuracy': 0.5091743119266054, 'eval_runtime': 2.8869, 'eval_samples_per_second': 302.056, 'eval_steps_per_second': 19.052}"
7,0.5091743119266054,"{'eval_loss': 152.88572692871094, 'eval_accuracy': 0.5091743119266054, 'eval_runtime': 2.831, 'eval_samples_per_second': 308.017, 'eval_steps_per_second': 19.428}"
2,0.5091743119266054,"{'eval_loss': 41.56632995605469, 'eval_model_preparation_time': 0.0033, 'eval_accuracy': 0.5091743119266054, 'eval_runtime': 2.2274, 'eval_samples_per_second': 391.488, 'eval_steps_per_second': 48.936}"
8,0.4908256880733945,"{'eval_loss': nan, 'eval_accuracy': 0.4908256880733945, 'eval_runtime': 2.8565, 'eval_samples_per_second': 305.264, 'eval_steps_per_second': 19.254}"
"""

# Read into DataFrame
from io import StringIO
df = pd.read_csv("/scratch/wd04/sm0074/fl_privacy/sst2_Linear_6_RDP.csv")

# Safely convert 'Info' column strings to dicts and extract eval_loss
def fix_and_extract_loss(info_str):
    try:
        fixed_str = info_str.replace("'", '"').replace("nan", "NaN")
        data = json.loads(fixed_str)
        return data.get("eval_loss", None)
    except json.JSONDecodeError:
        return None

df['eval_loss'] = df['Info'].apply(fix_and_extract_loss)
df = df.sort_values(by='Round')

# Plotting
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(df['Round'], df['Accuracy'], marker='o', label='Accuracy', color='blue')
plt.plot(df['Round'], df['eval_loss'], marker='x', label='Eval Loss', color='red')

plt.xlabel('Communication Round', fontsize=10)
plt.ylabel('Value', fontsize=10)
plt.title('Model Accuracy and Eval Loss Over Rounds', fontsize=11)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# Optional: Save the plot
plt.savefig("/scratch/wd04/sm0074/fl_privacy/figs/fl_accuracy_loss_plot.png", dpi=300)
plt.show()
