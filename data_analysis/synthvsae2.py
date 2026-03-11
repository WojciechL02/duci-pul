import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# ==========================================
# 0. File Paths & Data Loading
# ==========================================
model_name = "var_24"  # Replace with actual model name

paths = {
    "real_nonmem": f"../data/standard/{model_name}_llm_mia_cfg_real_nonmem_suspect_5k.npz",
    "ae_nonmem": f"../data/standard/{model_name}_llm_mia_cfg_5k_ae_nonmem_suspect_5k.npz",
    "syn_from_mem": f"../data/standard/{model_name}_llm_mia_cfg_5k_generated_nonmem_generated_from-mem_5k.npz",
    "syn_from_nonmem": f"../data/standard/{model_name}_llm_mia_cfg_5k_generated_nonmem_added_from-nonmem_5k.npz",
}


def load_and_reshape(path):
    data = np.load(path)["data"]
    return data.reshape(data.shape[0], -1)


# Load data
real_nonmem = load_and_reshape(paths["real_nonmem"])
ae_nonmem = load_and_reshape(paths["ae_nonmem"])

# Combine the two synthetic nonmember sources
syn_from_mem = load_and_reshape(paths["syn_from_mem"])
syn_from_nonmem = load_and_reshape(paths["syn_from_nonmem"])
synth_nonmem = np.vstack([syn_from_mem, syn_from_nonmem])

# ==========================================
# 1. PCA Density KDE & Wasserstein Distance
# ==========================================
print("Calculating PCA and Wasserstein Distances...")

# Fit PCA on the Synthetic Nonmembers to establish the target variance axis
pca = PCA(n_components=1)
syn_pc1 = pca.fit_transform(synth_nonmem).flatten()

# Project Real and AE Nonmembers onto the synthetic axis
real_pc1 = pca.transform(real_nonmem).flatten()
ae_pc1 = pca.transform(ae_nonmem).flatten()

# Calculate Wasserstein Distance (Earth Mover's Distance)
dist_real = wasserstein_distance(syn_pc1, real_pc1)
dist_ae = wasserstein_distance(syn_pc1, ae_pc1)

plt.figure(figsize=(10, 6))
sns.kdeplot(
    real_pc1,
    color="blue",
    fill=True,
    alpha=0.3,
    label=f"Real Nonmembers (W-Dist: {dist_real:.2f})",
)
sns.kdeplot(
    ae_pc1,
    color="green",
    fill=True,
    alpha=0.3,
    label=f"AE Nonmembers (W-Dist: {dist_ae:.2f})",
)
sns.kdeplot(
    syn_pc1,
    color="red",
    fill=False,
    linewidth=3,
    linestyle="--",
    label="Synthetic Nonmembers",
)

plt.title("Nonmember Distribution Shift on Principal Component 1")
plt.xlabel("Principal Component 1 Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 2. Domain Discriminator (Real vs Synth AND AE vs Synth)
# ==========================================
print("Training Domain Discriminators...")

# Setup 1: Can a classifier tell Real Nonmembers from Synthetic Nonmembers?
X_real_vs_syn = np.vstack([real_nonmem, synth_nonmem])
y_real_vs_syn = np.array([0] * len(real_nonmem) + [1] * len(synth_nonmem))

# Setup 2: Can a classifier tell AE Nonmembers from Synthetic Nonmembers?
X_ae_vs_syn = np.vstack([ae_nonmem, synth_nonmem])
y_ae_vs_syn = np.array([0] * len(ae_nonmem) + [1] * len(synth_nonmem))

clf = LogisticRegression(max_iter=1000)

# Get cross-validated probabilities
probs_real = cross_val_predict(
    clf, X_real_vs_syn, y_real_vs_syn, cv=3, method="predict_proba"
)[:, 1]
probs_ae = cross_val_predict(
    clf, X_ae_vs_syn, y_ae_vs_syn, cv=3, method="predict_proba"
)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Plot Before AE (Real vs Synth)
sns.histplot(
    probs_real[y_real_vs_syn == 0],
    color="blue",
    label="Real Nonmembers",
    kde=True,
    ax=axes[0],
    stat="density",
    alpha=0.5,
)
sns.histplot(
    probs_real[y_real_vs_syn == 1],
    color="red",
    label="Synth Nonmembers",
    kde=True,
    ax=axes[0],
    stat="density",
    alpha=0.5,
)
axes[0].set_title("Before AE: Easy to Separate\n(Real vs Synthetic)")
axes[0].set_xlabel("Probability of being Synthetic")
axes[0].legend()

# Plot After AE (AE vs Synth)
sns.histplot(
    probs_ae[y_ae_vs_syn == 0],
    color="green",
    label="AE Nonmembers",
    kde=True,
    ax=axes[1],
    stat="density",
    alpha=0.5,
)
sns.histplot(
    probs_ae[y_ae_vs_syn == 1],
    color="red",
    label="Synth Nonmembers",
    kde=True,
    ax=axes[1],
    stat="density",
    alpha=0.5,
)
axes[1].set_title("After AE: Classifier is Confused\n(AE vs Synthetic)")
axes[1].set_xlabel("Probability of being Synthetic")
axes[1].legend()

plt.tight_layout()
plt.show()
