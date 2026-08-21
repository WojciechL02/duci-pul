"""Power scan: run the 100x bootstrap DI at several sample sizes, to see at what
n the CLiD signal crosses p=0.05.

Writes a block to cdi/out_t2i_mia/results.md describing the scan.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

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


def _load(path):
    d = np.load(path, allow_pickle=True)["data"]
    # Some legacy feature dumps are float16 with values that overflow under fp16 reductions;
    # promote to float64 once on load so means/stds are stable.
    d = np.asarray(d).astype(np.float64)
    if d.ndim > 1:
        d = d.reshape(d.shape[0], -1)
    return d


def _score_channel(arr, channel):
    if arr.ndim == 1:
        return arr
    if channel is None:
        # project multi-channel CLiD features to a scalar by taking mean over alpha
        return arr.mean(axis=1)
    return arr[:, channel]


def bootstrap_one(members, nonmembers, n_per_side, n_iters, members_lower, seed):
    pvals = np.empty(n_iters)
    ok = np.empty(n_iters, dtype=bool)
    for i in range(n_iters):
        rng = np.random.default_rng(seed + i)
        mi = rng.choice(len(members), n_per_side, replace=False)
        ni = rng.choice(len(nonmembers), n_per_side, replace=False)
        p, correct = get_p_value(members[mi], nonmembers[ni], members_lower)
        pvals[i] = p
        ok[i] = correct
    return {
        "p_mean": float(pvals.mean()),
        "p_median": float(np.median(pvals)),
        "frac_p05": float((pvals < 0.05).mean()),
        "frac_correct": float(ok.mean()),
    }


def _parse_bool(s):
    return s.lower() in ("true", "1", "yes", "t", "y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--nonmembers", required=True)
    ap.add_argument("--channel", type=int, default=None,
                    help="which CLiD alpha channel to use; default: mean of all")
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[50, 100, 200, 500, 1000, 2000])
    ap.add_argument("--n-iters", type=int, default=100)
    ap.add_argument("--members-lower", type=_parse_bool, default=False)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="cdi/out_t2i_mia/results.md")
    args = ap.parse_args()

    m_raw = _load(args.members)
    v_raw = _load(args.nonmembers)
    m = _score_channel(m_raw, args.channel)
    v = _score_channel(v_raw, args.channel)
    print(f"[info] members pool={len(m)}  non-members pool={len(v)}  "
          f"channel={args.channel}  members_lower={args.members_lower}")
    print(f"[info] members mean±std {m.mean():.4f}±{m.std():.4f} | "
          f"non-members mean±std {v.mean():.4f}±{v.std():.4f}")

    lines = [
        f"## {args.label} — power scan",
        "",
        f"- timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"- members file: `{args.members}`",
        f"- non-members file: `{args.nonmembers}`",
        f"- channel: {'mean(alpha)' if args.channel is None else f'alpha[{args.channel}]'}",
        f"- members_lower: {args.members_lower}",
        f"- member pool: {len(m)} | non-member pool: {len(v)}",
        f"- members score mean±std: {m.mean():.4f} ± {m.std():.4f}",
        f"- non-members score mean±std: {v.mean():.4f} ± {v.std():.4f}",
        f"- Δ mean: {m.mean() - v.mean():+.4f}",
        "",
        "| n/side | p_mean | p_median | frac(p<0.05) | frac(correct dir) |",
        "|-------:|-------:|---------:|-------------:|------------------:|",
    ]
    for n in args.ns:
        if n > min(len(m), len(v)):
            continue
        st = bootstrap_one(m, v, n, args.n_iters, args.members_lower, 0)
        lines.append(
            f"| {n:>6d} | {st['p_mean']:.4f} | {st['p_median']:.4f} | "
            f"{st['frac_p05']:.0%} | {st['frac_correct']:.0%} |"
        )
        print(f"  n={n:5d}  p_mean={st['p_mean']:.4f}  frac(p<0.05)={st['frac_p05']:.0%}")
    lines.extend(["", "---", ""])
    block = "\n".join(lines)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if not os.path.exists(args.out):
        with open(args.out, "w") as fh:
            fh.write("# T2I MIA pilot — results\n\n---\n\n")
    with open(args.out, "a") as fh:
        fh.write(block)
    print(f"[info] appended results to {args.out}")


if __name__ == "__main__":
    main()
