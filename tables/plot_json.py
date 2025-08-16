import pandas as pd
import matplotlib.pyplot as plt
import json

# --- 1. Updated function to load data from a CSV file ---
# This function reads your CSV files.

raw = """{"round": 1, "client_id": 2, "noise_added": 0.0010299838202792524, "epsilon": 8.98890241172568, "delta": 0.01, "current_loss": 0.24966685473918915, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9997334837913514, "noise_multiplier": 0.0034332794009308415}
{"round": 1, "client_id": 1, "noise_added": 0.001036358125085002, "epsilon": 8.915941499112128, "delta": 0.01, "current_loss": 0.2574007213115692, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0059205770492554, "noise_multiplier": 0.0034545270836166737}
{"round": 1, "client_id": 3, "noise_added": 0.001034047261355658, "epsilon": 8.942235379224966, "delta": 0.01, "current_loss": 0.2545969784259796, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0036775827407838, "noise_multiplier": 0.00344682420451886}
{"round": 2, "client_id": 2, "noise_added": 0.0009087423042012613, "epsilon": 10.622196359951234, "delta": 0.01, "current_loss": 0.242384672164917, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9939077377319336, "noise_multiplier": 0.0030291410140042044}
{"round": 2, "client_id": 3, "noise_added": 0.0009130220732172843, "epsilon": 10.556904627157277, "delta": 0.01, "current_loss": 0.24823574721813202, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9985885977745057, "noise_multiplier": 0.0030434069107242814}
{"round": 2, "client_id": 0, "noise_added": 0.0009198985501399521, "epsilon": 10.451106450798457, "delta": 0.01, "current_loss": 0.2576369047164917, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0061095237731934, "noise_multiplier": 0.003066328500466507}
{"round": 3, "client_id": 2, "noise_added": 0.0008137932740650291, "epsilon": 12.324579937469057, "delta": 0.01, "current_loss": 0.24147100746631622, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.993176805973053, "noise_multiplier": 0.0027126442468834305}
{"round": 3, "client_id": 3, "noise_added": 0.0008156233141256532, "epsilon": 12.287407429453975, "delta": 0.01, "current_loss": 0.24426279962062836, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9954102396965028, "noise_multiplier": 0.0027187443804188444}
{"round": 3, "client_id": 0, "noise_added": 0.0008234877275046609, "epsilon": 12.130476366339527, "delta": 0.01, "current_loss": 0.2562602460384369, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0050081968307496, "noise_multiplier": 0.002744959091682203}
{"round": 4, "client_id": 0, "noise_added": 0.000743338487890711, "epsilon": 13.95401363925855, "delta": 0.01, "current_loss": 0.25282344222068787, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0022587537765504, "noise_multiplier": 0.0024777949596357035}
{"round": 4, "client_id": 2, "noise_added": 0.0007356780454892158, "epsilon": 14.151701731640658, "delta": 0.01, "current_loss": 0.23991252481937408, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9919300198554993, "noise_multiplier": 0.0024522601516307196}
{"round": 4, "client_id": 3, "noise_added": 0.0007367285440076522, "epsilon": 14.124226682778938, "delta": 0.01, "current_loss": 0.24168303608894348, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9933464288711549, "noise_multiplier": 0.0024557618133588407}
{"round": 5, "client_id": 0, "noise_added": 0.000678532817296639, "epsilon": 15.844101763651192, "delta": 0.01, "current_loss": 0.2509254813194275, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.000740385055542, "noise_multiplier": 0.0022617760576554634}
{"round": 5, "client_id": 3, "noise_added": 0.0006736767452938114, "epsilon": 16.0026136469057, "delta": 0.01, "current_loss": 0.24197295308113098, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9935783624649048, "noise_multiplier": 0.0022455891509793715}
{"round": 5, "client_id": 2, "noise_added": 0.0006732912058602236, "epsilon": 16.01509194490778, "delta": 0.01, "current_loss": 0.24126218259334564, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9930097460746765, "noise_multiplier": 0.0022443040195340787}
{"round": 6, "client_id": 3, "noise_added": 0.000621293388405458, "epsilon": 17.916906191563267, "delta": 0.01, "current_loss": 0.24073465168476105, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9925877213478089, "noise_multiplier": 0.002070977961351527}
{"round": 6, "client_id": 0, "noise_added": 0.0006267791351269445, "epsilon": 17.69369687013259, "delta": 0.01, "current_loss": 0.2516897916793823, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0013518333435059, "noise_multiplier": 0.0020892637837564816}
{"round": 6, "client_id": 2, "noise_added": 0.0006228536901048034, "epsilon": 17.85281853024126, "delta": 0.01, "current_loss": 0.24385060369968414, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9950804829597474, "noise_multiplier": 0.0020761789670160117}
{"round": 7, "client_id": 0, "noise_added": 0.0005851145019480536, "epsilon": 19.518468738021372, "delta": 0.01, "current_loss": 0.253933846950531, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.003147077560425, "noise_multiplier": 0.0019503816731601788}
{"round": 7, "client_id": 2, "noise_added": 0.0005807323911437898, "epsilon": 19.725066184074628, "delta": 0.01, "current_loss": 0.24454273283481598, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9956341862678528, "noise_multiplier": 0.0019357746371459661}
{"round": 7, "client_id": 3, "noise_added": 0.0005803218379106885, "epsilon": 19.744662145522177, "delta": 0.01, "current_loss": 0.24366289377212524, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9949303150177002, "noise_multiplier": 0.0019344061263689618}
{"round": 8, "client_id": 2, "noise_added": 0.0005478722230993405, "epsilon": 21.435626796313905, "delta": 0.01, "current_loss": 0.24889571964740753, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9991165757179261, "noise_multiplier": 0.0018262407436644684}
{"round": 8, "client_id": 3, "noise_added": 0.0005452730519031747, "epsilon": 21.58427709043512, "delta": 0.01, "current_loss": 0.242970809340477, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9943766474723816, "noise_multiplier": 0.001817576839677249}
{"round": 8, "client_id": 0, "noise_added": 0.0005488527154492741, "epsilon": 21.380098899578016, "delta": 0.01, "current_loss": 0.2511307895183563, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0009046316146852, "noise_multiplier": 0.0018295090514975805}
{"round": 9, "client_id": 0, "noise_added": 0.0005200236162004918, "epsilon": 23.146303950967233, "delta": 0.01, "current_loss": 0.25062254071235657, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0004980325698853, "noise_multiplier": 0.0017334120540016392}
{"round": 9, "client_id": 3, "noise_added": 0.0005160555110650747, "epsilon": 23.40950793451006, "delta": 0.01, "current_loss": 0.24107950925827026, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9928636074066163, "noise_multiplier": 0.0017201850368835823}
{"round": 9, "client_id": 2, "noise_added": 0.0005203936460225265, "epsilon": 23.121758268586312, "delta": 0.01, "current_loss": 0.25151243805885315, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0012099504470826, "noise_multiplier": 0.001734645486741755}
{"round": 10, "client_id": 2, "noise_added": 0.0004986481924468364, "epsilon": 24.581700630750227, "delta": 0.01, "current_loss": 0.2557733356952667, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 1.0046186685562135, "noise_multiplier": 0.0016621606414894549}
{"round": 10, "client_id": 0, "noise_added": 0.0004961976523169241, "epsilon": 24.756697436590187, "delta": 0.01, "current_loss": 0.24960200488567352, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9996816039085389, "noise_multiplier": 0.001653992174389747}
{"round": 10, "client_id": 3, "noise_added": 0.0004926441735710099, "epsilon": 25.015110895213567, "delta": 0.01, "current_loss": 0.24065308272838593, "base_noise": 0.1, "clipping_norm": 0.3, "loss_factor": 0.9925224661827088, "noise_multiplier": 0.0016421472452366997}"""

# Clean and parse line-by-line JSON objects (allow trailing commas on lines)
records = []
for line in raw.strip().splitlines():
    s = line.strip()
    if not s:
        continue
    # Remove trailing commas if present
    if s.endswith(','):
        s = s[:-1]
    # Parse JSON object
    try:
        obj = json.loads(s)
        records.append(obj)
    except json.JSONDecodeError as e:
        # Try to fix stray commas within the line (unlikely here)
        continue

# Create DataFrame
df = pd.DataFrame.from_records(records)

# Aggregate by round to handle duplicates (mean over clients per round)
agg = (
    df.groupby('round', as_index=False)
      .agg({
          'epsilon': 'mean',
          'noise_added': 'mean',
          'current_loss': 'mean',
          'noise_multiplier': 'mean'
      })
      .sort_values('round')
)

# --- Plotting (mirrors the style of the original script: subplots, grid, markers, tight_layout, savefig) ---
fig, axes = plt.subplots(4, 1, figsize=(8, 20))

plot_defs = [
    ("Epsilon vs Round", "epsilon", "Epsilon"),
    ("Noise Added vs Round", "noise_added", "Noise Added"),
    ("Current Loss vs Round", "current_loss", "Loss"),
    ("Noise Multiplier vs Round", "noise_multiplier", "Noise Multiplier"),
]

for i, (title, col, y_label) in enumerate(plot_defs):
    ax = axes[i]
    ax.plot(agg['round'], agg[col], label=y_label, marker='o')
    ax.set_title(title)
    ax.set_xlabel("Rounds")
    ax.set_ylabel(y_label)
    ax.set_xlim(left=0)
    ax.grid(True)
    ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = "./tables/plot/round_comparison_sst_iid.png"
plt.savefig(out_path)
# plt.show()


# # --- 6. Add a single, shared legend with original styling ---
# legend_labels = ["NoNoise", "FixedNoise", "DynamicNoise"]
# # Using the exact legend placement from your first script
# fig.legend(legend_labels, loc="outside center", ncol=3, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.01))

# # Adjust layout with original parameters
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# # --- 7. Save and show the final plot ---
# plt.savefig("noise_comparison_plot3.png")
# plt.show()
