import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ==========================================
# 0. File Paths & Data Loading
# ==========================================
model_name = "var_30"  # Replace with actual model name if dynamically formatting

paths = {
    "real_mem": f"../data/standard/{model_name}_llm_mia_cfg_real_mem_suspect_5k.npz",
    "real_nonmem": f"../data/standard/{model_name}_llm_mia_cfg_real_nonmem_suspect_5k.npz",
    "ae_mem": f"../data/standard/{model_name}_llm_mia_cfg_5k_ae_mem_suspect_5k.npz",
    "ae_nonmem": f"../data/standard/{model_name}_llm_mia_cfg_5k_ae_nonmem_suspect_5k.npz",
    "syn_from_mem": f"../data/standard/{model_name}_llm_mia_cfg_5k_generated_nonmem_generated_from-mem_5k.npz",
    "syn_from_nonmem": f"../data/standard/{model_name}_llm_mia_cfg_5k_generated_nonmem_added_from-nonmem_5k.npz",
}


def load_and_reshape(path):
    # Loads the npz, extracts 'data', and flattens [5000, 1, num_features] -> [5000, num_features]
    data = np.load(path)["data"]
    return data.reshape(data.shape[0], -1)


# Load data
real_mem = load_and_reshape(paths["real_mem"])
real_nonmem = load_and_reshape(paths["real_nonmem"])
ae_mem = load_and_reshape(paths["ae_mem"])
ae_nonmem = load_and_reshape(paths["ae_nonmem"])
syn_from_mem = load_and_reshape(paths["syn_from_mem"])
syn_from_nonmem = load_and_reshape(paths["syn_from_nonmem"])

# Grouping the datasets
unlabeled_real = np.vstack([real_mem, real_nonmem])
unlabeled_ae = np.vstack([ae_mem, ae_nonmem])
synthetic_negatives = np.vstack([syn_from_mem, syn_from_nonmem])

# Sample down for t-SNE performance if necessary (e.g., taking 1000 from each for cleaner plots)
# It's highly recommended to subsample for visualization clarity to avoid "overplotting"
n_samples = 1500
np.random.seed(42)

idx_unl = np.random.choice(unlabeled_real.shape[0], n_samples, replace=False)
idx_syn = np.random.choice(synthetic_negatives.shape[0], n_samples, replace=False)

unl_real_sub = unlabeled_real[idx_unl]
unl_ae_sub = unlabeled_ae[idx_unl]  # Use same indices to compare apples to apples
syn_sub = synthetic_negatives[idx_syn]

# ==========================================
# 1. Distribution Alignment Plot (t-SNE)
# ==========================================
print("Running t-SNE for Plot A (Before AE)...")
# Plot A Data: Real Unlabeled vs Raw Synthetics
X_plotA = np.vstack([unl_real_sub, syn_sub])
labels_plotA = np.array(
    ["Unlabeled (Real)"] * n_samples + ["Synthetic Negatives"] * n_samples
)
tsne_A = TSNE(n_components=2, random_state=42).fit_transform(X_plotA)

print("Running t-SNE for Plot B (After AE)...")
# Plot B Data: Autoencoded Unlabeled vs Raw Synthetics
# (Note: If you also autoencode the synthetics, replace syn_sub with ae_syn_sub here)
X_plotB = np.vstack([unl_ae_sub, syn_sub])
labels_plotB = np.array(
    ["Unlabeled (Autoencoded)"] * n_samples + ["Synthetic Negatives"] * n_samples
)
tsne_B = TSNE(n_components=2, random_state=42).fit_transform(X_plotB)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot A
axes[0].scatter(
    tsne_A[:n_samples, 0],
    tsne_A[:n_samples, 1],
    c="blue",
    label="Unlabeled (Real)",
    alpha=0.5,
    s=10,
)
axes[0].scatter(
    tsne_A[n_samples:, 0],
    tsne_A[n_samples:, 1],
    c="red",
    label="Synthetic Negatives",
    alpha=0.5,
    s=10,
)
axes[0].set_title("Plot A: Before Autoencoder (Distribution Shift)")
axes[0].legend()

# Plot B
axes[1].scatter(
    tsne_B[:n_samples, 0],
    tsne_B[:n_samples, 1],
    c="green",
    label="Unlabeled (AE Reconstructed)",
    alpha=0.5,
    s=10,
)
axes[1].scatter(
    tsne_B[n_samples:, 0],
    tsne_B[n_samples:, 1],
    c="red",
    label="Synthetic Negatives",
    alpha=0.5,
    s=10,
)
axes[1].set_title("Plot B: After Autoencoder (Distributions Aligned)")
axes[1].legend()

plt.tight_layout()
plt.show()

# ==========================================
# 2. Precision-Recall Curve (Ground Truth Check)
# ==========================================
print("Generating Precision-Recall Curve...")

# To show that your synthetic negatives actually help, we train a classifier on the
# synthetic negatives (as class 1, since in your PU setup they are the "Known Positives")
# and the Unlabeled set (as class 0).
# Then we evaluate it against the GROUND TRUTH of the unlabeled set (real_mem vs real_nonmem).

# --- Setup Training Data ---
# We'll use a subset of the autoencoded data for training to represent the "cleaned" state
X_train_unlabeled, X_test_unlabeled, y_train_unl_gt, y_test_unl_gt = train_test_split(
    unlabeled_ae,
    np.array(
        [1] * len(ae_mem) + [0] * len(ae_nonmem)
    ),  # Ground truth: 1 for mem, 0 for nonmem
    test_size=0.3,
    random_state=42,
)

# In PU learning, the classifier sees Unlabeled as '0' and Known Negatives (Synthetic) as '1'
X_train_pu = np.vstack([X_train_unlabeled, synthetic_negatives])
y_train_pu = np.array([0] * len(X_train_unlabeled) + [1] * len(synthetic_negatives))

# Train a proxy model (Logistic Regression is a good baseline to show linear separability)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_pu, y_train_pu)

# --- Evaluate on Ground Truth Test Set ---
# We want to see how well the model isolates the actual "suspects" (members vs non-members)
# within the unlabeled test set, using only the synthetic data as its guide.
y_scores = clf.predict_proba(X_test_unlabeled)[
    :, 1
]  # Probability of being a "non-member"

# Calculate Precision-Recall
precision, recall, thresholds = precision_recall_curve(
    ~y_test_unl_gt.astype(bool), y_scores
)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(
    recall, precision, color="purple", lw=2, label=f"PU Model (AUC = {pr_auc:.3f})"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Ground Truth Identification")
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.show()
