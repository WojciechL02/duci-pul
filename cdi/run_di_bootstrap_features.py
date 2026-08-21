"""Bootstrap DI directly on raw CLiD features (skip the LR classifier).

CLiD features are shape (B, 1, 11) — 11 alpha-mixtures of L_xc and D_xcci.
For each alpha channel, this runs the 100x(50/50) Welch bootstrap independently
and picks the alpha with the best mean p-value. Useful to check whether a weak
LR classifier is masking a real direct signal in the features themselves.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_CDI_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CDI_DIR)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_cdi_evaluate", os.path.join(_CDI_DIR, "src", "evaluation", "evaluate.py")
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[arg-type]
get_p_value = _mod.get_p_value


def _load_feats(path):
    d = np.load(path, allow_pickle=True)["data"]
    d = np.asarray(d).astype(np.float32)
    if d.ndim == 3:
        d = d[:, 0, :]  # (N, 1, 11) -> (N, 11)
    return d


def bootstrap_direction_free(members, nonmembers, n_iters=100, n_per_side=50, rng_seed=0):
    """For each feature channel and each sign, compute bootstrap p-value statistics.
    Returns the best (alpha_idx, members_lower, stats)."""
    best = None
    for a in range(members.shape[1]):
        m = members[:, a]
        v = nonmembers[:, a]
        for members_lower in (False, True):
            pvals = np.empty(n_iters)
            ok = np.empty(n_iters, dtype=bool)
            for i in range(n_iters):
                rng = np.random.default_rng(rng_seed + i)
                mi = rng.choice(len(m), n_per_side, replace=False)
                ni = rng.choice(len(v), n_per_side, replace=False)
                p, correct = get_p_value(m[mi], v[ni], members_lower)
                pvals[i] = p
                ok[i] = correct
            stats = {
                "alpha": a,
                "members_lower": members_lower,
                "p_mean": float(np.mean(pvals)),
                "p_median": float(np.median(pvals)),
                "frac_p05": float(np.mean(pvals < 0.05)),
                "frac_correct": float(np.mean(ok)),
            }
            if best is None or stats["p_mean"] < best["p_mean"]:
                best = stats
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--nonmembers", required=True)
    ap.add_argument("--n-iters", type=int, default=100)
    ap.add_argument("--n-per-side", type=int, default=50)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    m = _load_feats(args.members)
    v = _load_feats(args.nonmembers)
    print(f"[info] members {m.shape}, non-members {v.shape}")
    print(f"[info] per-alpha means:")
    for a in range(m.shape[1]):
        print(f"  alpha[{a:2d}]  members {m[:,a].mean():8.3f} ± {m[:,a].std():.3f}  "
              f"nonmembers {v[:,a].mean():8.3f} ± {v[:,a].std():.3f}  "
              f"delta={m[:,a].mean()-v[:,a].mean():+.4f}")
    best = bootstrap_direction_free(m, v, args.n_iters, args.n_per_side)
    print(f"\n# {args.label} (raw CLiD feature bootstrap)")
    print(
        f"  best alpha: {best['alpha']}, members_lower={best['members_lower']}\n"
        f"  p_mean   = {best['p_mean']:.4f}\n"
        f"  p_median = {best['p_median']:.4f}\n"
        f"  frac(p<0.05)     = {best['frac_p05']:.2%}\n"
        f"  frac(correct dir) = {best['frac_correct']:.2%}"
    )


if __name__ == "__main__":
    main()
