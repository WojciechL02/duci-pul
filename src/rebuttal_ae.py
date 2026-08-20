import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import prepare_data


def get_suspect_set(target, run_type, p, ss_len, data_dir, seed):
    """Return suspect-set (U) features and ground-truth membership labels."""
    X, y, s, _ = prepare_data(
        name=target,
        seed=seed,
        p=p,
        ss_len=ss_len,
        run_type=run_type,
        data_dir=data_dir,
    )
    return X[s == 0], y[s == 0]


def fit_axis(X, y, seed):
    """Membership classifier; returns unit weight vector and its norm."""
    clf = LogisticRegression(max_iter=5000, random_state=seed)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    n = np.linalg.norm(w)
    return w / n, n, clf


def cosine(u, v):
    return float(np.dot(u, v))  # both already unit norm


def run_once(target, p, ss_len, data_dir, seed):
    X_real, y_real = get_suspect_set(target, "synth", p, ss_len, data_dir, seed)
    X_ae, y_ae = get_suspect_set(target, "ae_synth", p, ss_len, data_dir, seed)

    assert X_real.shape == X_ae.shape, "suspect sets differ in shape"
    assert np.array_equal(y_real, y_ae), "membership labels not row-aligned"
    y = y_real.astype(int)

    # One coordinate system for both, so the coefficient vectors are comparable.
    scaler = StandardScaler().fit(X_real)
    Xr, Xa = scaler.transform(X_real), scaler.transform(X_ae)

    w_real, n_real, clf_real = fit_axis(Xr, y, seed)
    w_ae, n_ae, clf_ae = fit_axis(Xa, y, seed)
    cos_ae = cosine(w_real, w_ae)

    # Noise floor: two classifiers fit on disjoint halves of the SAME data.
    # cos_ae should be read against this, not against 1.0.
    i1, i2 = train_test_split(
        np.arange(len(y)), test_size=0.5, random_state=seed, stratify=y
    )
    w_h1, _, _ = fit_axis(Xr[i1], y[i1], seed)
    w_h2, _, _ = fit_axis(Xr[i2], y[i2], seed)
    cos_null = cosine(w_h1, w_h2)

    # Does the real-fit axis still separate the AE data?
    # If AE only shortens the margin, transfer should be near-lossless.
    auc_real_on_real = roc_auc_score(y, Xr @ w_real)
    auc_ae_on_ae = roc_auc_score(y, Xa @ w_ae)
    auc_real_on_ae = roc_auc_score(y, Xa @ w_real)

    return dict(
        cos_ae=cos_ae,
        cos_null=cos_null,
        norm_real=n_real,
        norm_ae=n_ae,
        norm_ratio=n_ae / n_real,
        auc_real_on_real=auc_real_on_real,
        auc_ae_on_ae=auc_ae_on_ae,
        auc_real_on_ae=auc_real_on_ae,
        transfer_loss=auc_ae_on_ae - auc_real_on_ae,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Does autoencoding rotate the membership-separating axis, "
        "or only shorten the margin along it?"
    )
    parser.add_argument("--target", type=str, default="var_30")
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--ss_len", type=int, default=2000)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="data/standard")
    args = parser.parse_args()

    runs = [
        run_once(args.target, args.prob, args.ss_len, args.data_dir, seed)
        for seed in range(args.n_runs)
    ]
    g = lambda k: np.array([r[k] for r in runs])

    print(f"\ntarget={args.target}  p={args.prob}  |U|={args.ss_len}  seeds={args.n_runs}")
    print("-" * 62)
    print(f"cos(w_real, w_AE)          {g('cos_ae').mean():.4f} +/- {g('cos_ae').std():.4f}")
    print(f"cos(w_half1, w_half2)      {g('cos_null').mean():.4f} +/- {g('cos_null').std():.4f}"
          "   <- sampling-noise floor")
    print(f"||w_AE|| / ||w_real||      {g('norm_ratio').mean():.4f} +/- {g('norm_ratio').std():.4f}")
    print("-" * 62)
    print(f"AUC  real axis on real     {g('auc_real_on_real').mean():.4f}")
    print(f"AUC  AE   axis on AE       {g('auc_ae_on_ae').mean():.4f}")
    print(f"AUC  real axis on AE       {g('auc_real_on_ae').mean():.4f}")
    print(f"transfer loss              {g('transfer_loss').mean():.4f}"
          "   <- ~0 means the axis is unchanged")


if __name__ == "__main__":
    main()
