""" Generate 1,000-trial synthetic EEG dataset, 4 classes: rest, left, right, feet, Save to data/raw/dataset.npz """

import numpy as np 
from pathlib import Path
from src.generate_eeg import SyntheticEEG
import matplotlib.pyplot as plt

# config
N_TRIALS_PER_CLASS = 250 
CLASSES = ["rest", "left", "right", "feet"]
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# initialize generator
eeg_gen = SyntheticEEG()

# storage
X = [] # EEG trials: list of (8, 1000)
y = [] # Labels: 0=rest, 1=left, 2=right, 3=feet

print('Generating dataset...')
for label in CLASSES:
    class_idx = CLASSES.index(label)
    print(f" {label}: {N_TRIALS_PER_CLASS} trials")
    for _ in range(N_TRIALS_PER_CLASS):
        trial = eeg_gen.generate_trial(label)
        X.append(trial)
        y.append(class_idx)

# convert to arrays 
X = np.array(X) # (1000, 8, 1000)
y = np.array(y) # (1000, )

# visualize
from scipy.signal import welch

def compute_mu_power(trial, fs=250):
    power = 0 
    for ch in [0, 1, 2]: # C#, Cz, C4
        f, Pxx = welch(trial[ch], fs=fs, nperseg=500)
        mu_mask = (f >= 8) & (f <= 12)
        power += np.sum(Pxx[mu_mask])
    return power / 3

mu_powers = {"rest": [], "left": [], "right": [], "feet": []}
for trial, label in zip(X, y):
    mu_powers[CLASSES[label]].append(compute_mu_power(trial))

# plot
plt.figure(figsize=(10, 6))
for label in CLASSES:
    plt.hist(mu_powers[label], alpha=0.7, label=label, bins=30)
plt.title("Mu Band Power Distribution by Class")
plt.xlabel("Average Power (8-12 Hz)")
plt.ylabel("Trial Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/mu_power_distribution.png", dpi=150)
plt.show()

print("Visualization saved: docs/mu_power_distribution.png")
