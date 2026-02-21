import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def evaluate_pul_feasibility(X_real, y_real, X_synth, p_label=1):
    """
    Evaluates if the synthetic dataset (X_synth) is a good Positive Component
    for PUL training on the Real dataset (X_real).

    Args:
        X_real (np.ndarray): Shape [n_samples, n_features]. The mixed/unlabeled data.
        y_real (np.ndarray): Shape [n_samples,]. Ground truth labels for X_real.
                             (0 = Negative/Background, 1 = Positive/Target)
        X_synth (np.ndarray): Shape [m_samples, n_features]. The pure positive component.
        p_label (int): The label in y_real that corresponds to the positive class (nonmem).
    """

    # 1. Input Validation
    assert isinstance(X_real, np.ndarray), "X_real must be a numpy array"
    assert isinstance(X_synth, np.ndarray), "X_synth must be a numpy array"
    assert X_real.ndim == 2, "X_real must be shape [n_samples, n_features]"
    assert X_synth.ndim == 2, "X_synth must be shape [n_samples, n_features]"
    assert (
        X_real.shape[1] == X_synth.shape[1]
    ), "Feature dimension mismatch between Real and Synth"

    # Flatten y if it's a column vector
    y_real = y_real.ravel()

    # Extract the "Real Positive" (nonmem) and "Real Negative" (mem) parts
    X_real_p = X_real[y_real == p_label]
    X_real_n = X_real[y_real != p_label]

    print(f"--- Data Statistics ---")
    print(f"Real Data (Total):   {X_real.shape}")
    print(f"  - Real Pos (nonmem): {X_real_p.shape[0]}")
    print(f"  - Real Neg (mem):    {X_real_n.shape[0]}")
    print(f"Synth Data (nonmem): {X_synth.shape}")
    print("-" * 30)

    # ---------------------------------------------------------
    # TEST 1: Representativeness (Covariate Shift)
    # ---------------------------------------------------------
    # Goal: Can a classifier distinguish Synth vs Real Positives?
    # Ideal AUC: 0.5 (Indistinguishable)

    print("\n[Test 1] Covariate Shift Check (Synth vs Real Positive)")

    # Create dataset: Class 0 = Real Pos, Class 1 = Synth
    X_shift = np.vstack((X_real_p, X_synth))
    y_shift = np.hstack((np.zeros(len(X_real_p)), np.ones(len(X_synth))))

    # Use a robust classifier (Random Forest)
    clf = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )

    # 5-Fold CV
    if len(X_shift) < 50:
        print("  ! Not enough data for reliable Covariate Shift test.")
    else:
        shift_auc = np.mean(
            cross_val_score(clf, X_shift, y_shift, cv=5, scoring="roc_auc")
        )

        print(f"  Classifier AUC: {shift_auc:.4f}")
        if shift_auc < 0.6:
            print(
                "  >> RESULT: EXCELLENT. Synth data is very similar to Real Positives."
            )
        elif shift_auc < 0.8:
            print("  >> RESULT: ACCEPTABLE. Some shift detected, but likely usable.")
        else:
            print(
                "  >> RESULT: DANGER. Synth data looks significantly different from Real Positives."
            )

    # ---------------------------------------------------------
    # TEST 2: Separability (Real Pos vs Real Neg)
    # ---------------------------------------------------------
    # Goal: Do the features contain enough signal to separate classes?
    # Ideal AUC: > 0.7 (Separable)

    print("\n[Test 2] Feature Separability Check (Real Pos vs Real Neg)")

    clf_sep = RandomForestClassifier(
        n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
    )
    sep_auc = np.mean(cross_val_score(clf_sep, X_real, y_real, cv=5, scoring="roc_auc"))

    print(f"  Classifier AUC: {sep_auc:.4f}")
    if sep_auc > 0.75:
        print("  >> RESULT: GOOD. Features are discriminative.")
    elif sep_auc > 0.6:
        print("  >> RESULT: WEAK. Features are noisy.")
    else:
        print(
            "  >> RESULT: POOR. Classes are overlapping or features are non-informative."
        )

    # ---------------------------------------------------------
    # TEST 3: Visual Inspection (Projection)
    # ---------------------------------------------------------
    print("\n[Test 3] Visualizing Distributions...")

    # Downsample for visualization if datasets are massive (>2000 points)
    n_limit = 1000

    # Helper to safe sample
    def safe_sample(arr, n):
        if len(arr) > n:
            idx = np.random.choice(len(arr), n, replace=False)
            return arr[idx]
        return arr

    X_rp_sub = safe_sample(X_real_p, n_limit)
    X_rn_sub = safe_sample(X_real_n, n_limit)
    X_sy_sub = safe_sample(X_synth, n_limit)

    X_vis = np.vstack((X_rp_sub, X_rn_sub, X_sy_sub))

    # Create label array for coloring
    # 0: Real Neg (Mem), 1: Real Pos (NonMem), 2: Synth (NonMem)
    y_vis = np.concatenate(
        [np.zeros(len(X_rn_sub)), np.ones(len(X_rp_sub)), np.full(len(X_sy_sub), 2)]
    )

    # Standardize before projection
    X_vis_scaled = StandardScaler().fit_transform(X_vis)

    # Compute t-SNE
    print("  Running t-SNE...")
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
    X_emb = tsne.fit_transform(X_vis_scaled)

    # Plot
    plt.figure(figsize=(10, 7))

    # Plot Real Neg (Red)
    plt.scatter(
        X_emb[y_vis == 0, 0],
        X_emb[y_vis == 0, 1],
        c="red",
        alpha=0.3,
        label="Real Neg (Mem)",
        s=20,
    )

    # Plot Real Pos (Green)
    plt.scatter(
        X_emb[y_vis == 1, 0],
        X_emb[y_vis == 1, 1],
        c="green",
        alpha=0.5,
        label="Real Pos (NonMem)",
        s=20,
    )

    # Plot Synth (Blue)
    plt.scatter(
        X_emb[y_vis == 2, 0],
        X_emb[y_vis == 2, 1],
        c="blue",
        alpha=0.5,
        marker="x",
        label="Synth (NonMem)",
        s=30,
    )

    plt.title("PUL Data Check: t-SNE Projection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


from scipy.stats import ks_2samp
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


def run_advanced_diagnostics(X_real, y_real, X_synth, p_label=1):
    # Setup Data
    X_real_p = X_real[y_real == p_label]
    X_real_n = X_real[y_real != p_label]

    true_alpha = len(X_real_p) / len(X_real)
    print(f"\n--- ADVANCED DIAGNOSTICS ---")
    print(f"True Alpha (Class Prior) in Real: {true_alpha:.4f}")

    # ---------------------------------------------------------
    # TEST 4: Feature-wise KS Test (Drift Detection)
    # ---------------------------------------------------------
    print("\n[Test 4] Feature-wise Drift (KS Test)")

    n_features = X_real.shape[1]
    drifting_features = []

    # We use a strict p-value threshold (Bonferroni correction could be applied)
    # But for a quick check, p < 0.01 is a good "drift" indicator.
    threshold = 0.05

    print(f"  {'FeatIdx':<10} {'KS-Stat':<10} {'P-Value':<10} {'Status'}")
    print(f"  {'-' * 40}")

    for i in range(n_features):
        stat, p_val = ks_2samp(X_real_p[:, i], X_synth[:, i])

        status = "DRIFT" if p_val < 0.001 else "OK"  # Strict threshold
        if status == "DRIFT":
            drifting_features.append(i)

        # Print top 5 and any drifters
        if i < 5 or status == "DRIFT":
            print(f"  {i:<10} {stat:.4f}     {p_val:.2e}     {status}")

    if len(drifting_features) > n_features * 0.5:
        print(f"  >> WARNING: Over 50% of features show significant statistical drift.")
    else:
        print(f"  >> GOOD: Most features follow the same statistical distribution.")

    # ---------------------------------------------------------
    # TEST 5: Alpha Recovery (Can we estimate the prior?)
    # ---------------------------------------------------------
    print("\n[Test 5] Alpha Recovery Check (Using KM2/Ratio Logic)")

    # Simple Ratio Estimator logic (Simplified KM2)
    # 1. Train probabilistic classifier: Synth (P) vs Real (U)
    # Note: In standard PUL, we train P vs U.
    # Ideally, Prob(y=P|x) should help us recover alpha.

    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    # Setup P vs U training
    X_train = np.vstack([X_real, X_synth])
    # Label 0=U, 1=P (Synth) for the classifier training
    y_train = np.hstack([np.zeros(len(X_real)), np.ones(len(X_synth))])

    # Train Calibrated Classifier
    base_clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    # sigmoid calibration is usually safer for small datasets
    clf = CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)

    # Predict on U (Real)
    probs_u = clf.predict_proba(X_real)[:, 1]  # P(s=1|x)

    # Estimate Alpha: alpha = mean(P(y=1|x))
    # Using the invariance: P(s=1|x) = P(s=1) * P(x|s=1) / P(x)
    # A robust estimator is "Avg Probability of being Synth" / "Prior of Synth in Training"
    # But a simpler heuristic for 'feasibility':
    # If the classifier relies on features that distinguish P from U perfectly,
    # probabilities on U will be 0. This means alpha_hat -> 0.

    # Let's use the standard "deductive" estimator: sum(scores_U) / sum(scores_P) approx alpha?
    # Actually, let's use the standard KM2 baseline:
    # We look at the 'tails'. If prob is high, it's likely positive.

    # Heuristic: The average predicted probability on Real data
    # adjusted by the training ratio.
    train_prior = len(X_synth) / len(X_train)
    avg_score_u = np.mean(probs_u)

    # This is a rough 'scaled' estimator often used for sanity checks
    # If distributions match: P(s=1|x) is proportional to density ratio
    estimated_alpha_rough = avg_score_u

    print(f"  True Alpha:      {true_alpha:.3f}")
    print(f"  Recovered Alpha: {estimated_alpha_rough:.3f} (Rough estimate)")

    err = abs(true_alpha - estimated_alpha_rough)
    if err < 0.1:
        print("  >> RESULT: EXCELLENT. The logic recovers the class prior well.")
    elif err < 0.2:
        print("  >> RESULT: OKAY. Estimation is noisy but in the ballpark.")
    else:
        print(
            "  >> RESULT: FAIL. The estimator cannot see the positives in the real data."
        )

    # ---------------------------------------------------------
    # TEST 6: Mode Coverage (One-Class SVM)
    # ---------------------------------------------------------
    print("\n[Test 6] Mode Coverage (Mode Collapse Check)")

    # Train OCSVM on Synth
    # nu=0.05 means we allow 5% of Synth to be considered outliers (noise)
    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)

    # Scale data (SVM is sensitive to scaling)
    scaler = StandardScaler()
    X_synth_s = scaler.fit_transform(X_synth)
    X_real_p_s = scaler.transform(X_real_p)

    ocsvm.fit(X_synth_s)

    # Predict on Real Positives
    # 1 = Inlier, -1 = Outlier
    preds = ocsvm.predict(X_real_p_s)
    recall = np.mean(preds == 1)

    print(f"  Recall on Real Positives: {recall:.2%}")

    if recall > 0.85:
        print("  >> RESULT: GOOD. Synth covers most real positive variations.")
    elif recall > 0.50:
        print(
            "  >> RESULT: WARNING. Synth misses significant parts of the real distribution (Mode Collapse)."
        )
    else:
        print(
            "  >> RESULT: CRITICAL. Synth data is too narrow or disjoint from reality."
        )


if __name__ == "__main__":
    # model_name = "var_30_llm_mia_cfg"
    model_name = "uvit_t2i_deep_clid_5k"
    path1 = f"../data/standard/{model_name}_real_mem_suspect{'_5k' if 'dit' not in model_name else ''}.npz"
    path2 = f"../data/standard/{model_name}_real_nonmem_suspect{'_5k' if 'dit' not in model_name else ''}.npz"
    path3 = f"../data/standard/{model_name}_generated_nonmem_generated_from-mem{'_5k' if 'dit' not in model_name else ''}.npz"
    path4 = f"../data/standard/{model_name}_generated_nonmem_added_from-nonmem{'_5k' if 'dit' not in model_name else ''}.npz"
    max_data = 2000
    real_mem = np.load(path1, allow_pickle=True)["data"].squeeze(1)[: max_data // 2]
    real_nonmem = np.load(path2, allow_pickle=True)["data"].squeeze(1)[: max_data // 2]
    synth1 = np.load(path3, allow_pickle=True)["data"].squeeze(1)[: max_data // 2]
    synth2 = np.load(path4, allow_pickle=True)["data"].squeeze(1)[: max_data // 2]
    X_real = np.concatenate([real_mem, real_nonmem], axis=0)
    y_real = np.concatenate(
        [np.zeros(len(real_mem)), np.ones(len(real_nonmem))], axis=0
    )
    X_synth = np.concatenate([synth1, synth2], axis=0)

    evaluate_pul_feasibility(X_real, y_real, X_synth, p_label=1)
    run_advanced_diagnostics(X_real, y_real, X_synth)
