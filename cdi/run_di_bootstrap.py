"""
Bootstrap dataset-inference p-value from CLiD (or any other) scores.

Loads two .npz score files produced by `+action=scores_computation +attack=<attack>`
(one for members, one for non-members), then repeats `n_iters` times:
  - draw `n_per_side` random rows from each with a per-iteration seed
  - compute Welch t-test p-value and whether the direction matches `members_lower`
Report mean / median / std / IQR and fraction of iterations with p < alpha.

Usage:
    python cdi/run_di_bootstrap.py \
        --members cdi/out_t2i_mia/scores/<model>_clid_<run>_<ds>_train.npz \
        --nonmembers cdi/out_t2i_mia/scores/<model>_clid_<run>_<ds>_val.npz \
        --n-iters 100 --n-per-side 50 \
        --members-lower false \
        --label "<model> / <dataset>" \
        --out cdi/out_t2i_mia/results.md
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np

# Allow running the script either as `python cdi/run_di_bootstrap.py` from repo root
# or from inside cdi/. Just add the cdi dir so `src.evaluation.evaluate` resolves.
_CDI_DIR = os.path.dirname(os.path.abspath(__file__))
if _CDI_DIR not in sys.path:
    sys.path.insert(0, _CDI_DIR)

# Import evaluate.py directly as a standalone module to avoid pulling in the
# full cdi/src/__init__.py (which imports every wrapper + vendored lib).
import importlib.util as _ilu  # noqa: E402

_EVAL_PATH = os.path.join(_CDI_DIR, "src", "evaluation", "evaluate.py")
_spec = _ilu.spec_from_file_location("_cdi_evaluate", _EVAL_PATH)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[arg-type]
get_p_value = _mod.get_p_value


def _load_scores(path: str) -> np.ndarray:
    f = np.load(path, allow_pickle=True)
    data = f["data"]
    arr = np.asarray(data).astype(np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    mask = np.isfinite(arr)
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"[warn] dropping {dropped} non-finite entries in {path}")
        arr = arr[mask]
    return arr


def _pstr(x: float) -> str:
    if x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def bootstrap(
    members: np.ndarray,
    nonmembers: np.ndarray,
    n_iters: int,
    n_per_side: int,
    members_lower: bool,
    rng_seed: int = 0,
) -> dict:
    if len(members) < n_per_side:
        raise ValueError(f"members has {len(members)} samples < n_per_side={n_per_side}")
    if len(nonmembers) < n_per_side:
        raise ValueError(f"nonmembers has {len(nonmembers)} samples < n_per_side={n_per_side}")

    rng = np.random.default_rng(rng_seed)

    pvalues = np.empty(n_iters, dtype=np.float64)
    correct_order = np.empty(n_iters, dtype=bool)

    for i in range(n_iters):
        iter_rng = np.random.default_rng(rng_seed + i)
        m_idx = iter_rng.choice(len(members), size=n_per_side, replace=False)
        n_idx = iter_rng.choice(len(nonmembers), size=n_per_side, replace=False)
        p, ok = get_p_value(members[m_idx], nonmembers[n_idx], members_lower)
        pvalues[i] = p
        correct_order[i] = ok

    alpha = 0.05
    return {
        "n_iters": n_iters,
        "n_per_side": n_per_side,
        "members_lower": members_lower,
        "n_members_pool": int(len(members)),
        "n_nonmembers_pool": int(len(nonmembers)),
        "p_mean": float(np.mean(pvalues)),
        "p_median": float(np.median(pvalues)),
        "p_std": float(np.std(pvalues)),
        "p_q25": float(np.quantile(pvalues, 0.25)),
        "p_q75": float(np.quantile(pvalues, 0.75)),
        "p_min": float(np.min(pvalues)),
        "p_max": float(np.max(pvalues)),
        "frac_p_lt_alpha": float(np.mean(pvalues < alpha)),
        "frac_correct_order": float(np.mean(correct_order)),
        "frac_p_lt_and_correct": float(np.mean((pvalues < alpha) & correct_order)),
        "members_mean": float(np.mean(members)),
        "members_std": float(np.std(members)),
        "nonmembers_mean": float(np.mean(nonmembers)),
        "nonmembers_std": float(np.std(nonmembers)),
        "pvalues": pvalues,  # included for debugging; not written to results.md
    }


def success_decision(stats: dict, alpha: float = 0.05) -> str:
    if stats["p_mean"] < alpha and stats["frac_p_lt_alpha"] >= 0.5:
        return "SUCCESS"
    return "FAIL"


def render_block(label: str, members_path: str, nonmembers_path: str, stats: dict) -> str:
    decision = success_decision(stats)
    lines = [
        f"## {label}",
        "",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- members file: `{members_path}`",
        f"- non-members file: `{nonmembers_path}`",
        f"- members_lower: `{stats['members_lower']}`",
        f"- member pool size: {stats['n_members_pool']} | non-member pool size: {stats['n_nonmembers_pool']}",
        f"- bootstrap: {stats['n_iters']}x of {stats['n_per_side']} members vs {stats['n_per_side']} non-members",
        "",
        f"- members score mean ± std:     {stats['members_mean']:.4f} ± {stats['members_std']:.4f}",
        f"- non-members score mean ± std: {stats['nonmembers_mean']:.4f} ± {stats['nonmembers_std']:.4f}",
        "",
        f"- p-value mean:   **{_pstr(stats['p_mean'])}**",
        f"- p-value median: {_pstr(stats['p_median'])}",
        f"- p-value std:    {stats['p_std']:.4f}",
        f"- p-value IQR:    [{_pstr(stats['p_q25'])}, {_pstr(stats['p_q75'])}]",
        f"- p-value range:  [{_pstr(stats['p_min'])}, {_pstr(stats['p_max'])}]",
        f"- fraction p < 0.05:                       {stats['frac_p_lt_alpha']:.2%}",
        f"- fraction with correct direction:         {stats['frac_correct_order']:.2%}",
        f"- fraction with p < 0.05 AND correct dir:  {stats['frac_p_lt_and_correct']:.2%}",
        "",
        f"**Decision: {decision}** (mean p < 0.05 and ≥ 50% iters below 0.05)",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _parse_bool(x: str) -> bool:
    v = x.strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"not a bool: {x}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="path to members scores .npz")
    ap.add_argument("--nonmembers", required=True, help="path to non-members scores .npz")
    ap.add_argument("--n-iters", type=int, default=100)
    ap.add_argument("--n-per-side", type=int, default=50)
    ap.add_argument("--members-lower", type=_parse_bool, default=False,
                    help="whether members have lower scores (CLiD default: false)")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--label", required=True, help="markdown section label, e.g. 'minfm / imagenet'")
    ap.add_argument("--out", default="cdi/out_t2i_mia/results.md",
                    help="append results to this markdown file")
    args = ap.parse_args()

    members = _load_scores(args.members)
    nonmembers = _load_scores(args.nonmembers)

    print(f"[info] members pool={len(members)}, non-members pool={len(nonmembers)}")
    print(f"[info] bootstrap {args.n_iters}x of {args.n_per_side} + {args.n_per_side}")

    stats = bootstrap(
        members=members,
        nonmembers=nonmembers,
        n_iters=args.n_iters,
        n_per_side=args.n_per_side,
        members_lower=args.members_lower,
        rng_seed=args.rng_seed,
    )

    block = render_block(args.label, args.members, args.nonmembers, stats)
    print(block)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        header = (
            "# T2I MIA pilot — results\n\n"
            f"Created {datetime.now().isoformat(timespec='seconds')}. "
            "Each block below is one `(model, dataset)` pair with a 100x bootstrap DI test.\n\n"
            "---\n\n"
        )
        with open(args.out, "w") as fh:
            fh.write(header)
    with open(args.out, "a") as fh:
        fh.write(block)

    print(f"[info] appended results to {args.out}")


if __name__ == "__main__":
    main()
