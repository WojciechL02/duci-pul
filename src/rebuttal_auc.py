import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data import prepare_data
from utils import seed_everything


def get_suspect_set(target, run_type, p, ss_len, data_dir, seed):
    """Return only the suspect-set (U) rows, i.e. s_test == 0, dropping the N rows."""
    X, _, s, _ = prepare_data(
        name=target,
        seed=seed,
        p=p,
        ss_len=ss_len,
        run_type=run_type,
        data_dir=data_dir,
    )
    return X[s == 0]


def run_once(target, p, ss_len, data_dir, seed):
    X_suspect = get_suspect_set(target, "synth", p, ss_len, data_dir, seed)
    X_suspect_ae = get_suspect_set(target, "ae_synth", p, ss_len, data_dir, seed)

    X = np.concatenate([X_suspect, X_suspect_ae])
    label = np.concatenate(
        [np.zeros(len(X_suspect), dtype=int), np.ones(len(X_suspect_ae), dtype=int)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, label, test_size=0.2, random_state=seed, stratify=label
    )
    seed_everything(seed)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, prob)


def main():
    parser = argparse.ArgumentParser(
        description="AUC distinguishing X_suspect (real) from X_suspect_AE (rebuttal)."
    )
    parser.add_argument("--target", type=str, default="var_30")
    parser.add_argument("--prob", type=float, default=0.5)
    parser.add_argument("--ss_len", type=int, default=2000)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="../data/standard")
    args = parser.parse_args()

    aucs = [
        run_once(args.target, args.prob, args.ss_len, args.data_dir, seed)
        for seed in range(args.n_runs)
    ]
    print(
        f"X_suspect vs X_suspect_AE: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  (runs: {[f'{a:.4f}' for a in aucs]})"
    )


if __name__ == "__main__":
    main()
