import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

import umap  # uncomment to use UMAP instead of PCA

# ==========================================
# 0. Load Data (assuming same paths as before)
# ==========================================
model_name = "var_24"


def load_and_reshape(path):
    data = np.load(path)["data"]
    return data.reshape(data.shape[0], -1)


real_nonmem = load_and_reshape(
    f"../data/standard/{model_name}_llm_mia_cfg_real_nonmem_suspect_5k.npz"
)
ae_nonmem = load_and_reshape(
    f"../data/standard/{model_name}_llm_mia_cfg_5k_ae_nonmem_suspect_5k.npz"
)
syn_from_mem = load_and_reshape(
    f"../data/standard/{model_name}_llm_mia_cfg_5k_generated_nonmem_generated_from-mem_5k.npz"
)
syn_from_nonmem = load_and_reshape(
    f"../data/standard/{model_name}_llm_mia_cfg_5k_generated_nonmem_added_from-nonmem_5k.npz"
)
synth_nonmem = np.vstack([syn_from_mem, syn_from_nonmem])

# Combine for a shared projection
all_data = np.vstack([real_nonmem, ae_nonmem, synth_nonmem])
n_real, n_ae = len(real_nonmem), len(ae_nonmem)

# ==========================================
# 1. Dimensionality Reduction
# ==========================================
# Using PCA here, but swap to umap.UMAP() if the linear shift isn't obvious enough
# reducer = PCA(n_components=2)
reducer = umap.UMAP(n_components=2)
all_2d = reducer.fit_transform(all_data)

real_2d = all_2d[:n_real]
ae_2d = all_2d[n_real : n_real + n_ae]
synth_2d = all_2d[n_real + n_ae :]

# Calculate Centroids (Center of Mass)
centroid_real = np.mean(real_2d, axis=0)
centroid_ae = np.mean(ae_2d, axis=0)
centroid_synth = np.mean(synth_2d, axis=0)

# ==========================================
# 2. The Single "Shift Vector" Plot
# ==========================================
plt.figure(figsize=(10, 8))

# 1. The Target: Synthetic Nonmembers (Unfilled contours, thick dashed red lines)
sns.kdeplot(
    x=synth_2d[:, 0],
    y=synth_2d[:, 1],
    color="red",
    linewidths=2,
    linestyles="dashed",
    levels=4,
    thresh=0.1,
)

# 2. The Baseline: Real Nonmembers (Filled, transparent blue)
sns.kdeplot(
    x=real_2d[:, 0],
    y=real_2d[:, 1],
    color="blue",
    fill=True,
    alpha=0.3,
    levels=4,
    thresh=0.1,
)

# 3. The Solution: AE Nonmembers (Filled, transparent green)
sns.kdeplot(
    x=ae_2d[:, 0],
    y=ae_2d[:, 1],
    color="green",
    fill=True,
    alpha=0.4,
    levels=4,
    thresh=0.1,
)

# --- Plot Centroids and Shift Arrow ---
# Mark the centers
plt.scatter(*centroid_real, color="darkblue", s=100, marker="X", zorder=5)
plt.scatter(*centroid_ae, color="darkgreen", s=100, marker="X", zorder=5)
plt.scatter(*centroid_synth, color="darkred", s=100, marker="X", zorder=5)

# Draw an arrow showing the shift from Real to AE
plt.annotate(
    "",
    xy=centroid_ae,
    xytext=centroid_real,
    arrowprops=dict(facecolor="black", width=2, headwidth=10, shrink=0.05),
    zorder=4,
)

# --- Clean up and Legends ---
plt.title(
    "Autoencoder Reduction of Distribution Shift\n(Real → AE → Synthetic Target)",
    fontsize=14,
)
plt.xlabel("Component 1")
plt.ylabel("Component 2")

# Manual legend to match our custom styles
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="blue", alpha=0.3, label="Real Nonmembers (Original)"),
    Patch(facecolor="green", alpha=0.4, label="AE Nonmembers (Shifted)"),
    Line2D(
        [0],
        [0],
        color="red",
        lw=2,
        linestyle="dashed",
        label="Synthetic Nonmembers (Target)",
    ),
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="black",
        markersize=10,
        label="Distribution Centroids",
    ),
]
plt.legend(handles=legend_elements, loc="upper right", frameon=True, shadow=True)

plt.tight_layout()
plt.show()
