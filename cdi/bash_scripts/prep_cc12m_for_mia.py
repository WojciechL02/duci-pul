#!/usr/bin/env python3
"""Extract CC12M (pixparse/cc12m-wds) tar shards into a members/non-members
split using Nitro-T's own training filter: min(H,W) >= 512 is a member
(part of AMD's Nitro-T training slice), min(H,W) < 512 is a non-member
(rejected by AMD's data-prep, same URL corpus).

Each member/non-member is saved as:
    <out>/members/<subdir>/<uid>.jpg   + <uid>.txt   (caption)
    <out>/nonmembers/<subdir>/<uid>.jpg + <uid>.txt

subdir = first 2 chars of uid so we avoid too many entries in one directory.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import tarfile
from collections import defaultdict

from PIL import Image


def iter_wds_samples(tar_path):
    with tarfile.open(tar_path, "r") as tf:
        current = {}
        current_key = None
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            base, ext = os.path.splitext(name)
            ext = ext.lower().lstrip(".")
            if current_key is None or base != current_key:
                if current_key is not None and current:
                    yield current_key, current
                current = {}
                current_key = base
            f = tf.extractfile(member)
            if f is None:
                continue
            current[ext] = f.read()
        if current_key is not None and current:
            yield current_key, current


IMG_EXTS = ("jpg", "jpeg", "png", "webp")


def extract_caption(sample: dict):
    """Find a text caption in a WDS entry regardless of naming convention."""
    for key in ("txt", "caption", "txt.txt"):
        if key in sample:
            return sample[key].decode("utf-8", errors="replace").strip()
    if "json" in sample:
        try:
            j = json.loads(sample["json"])
        except Exception:
            return ""
        for k in ("caption", "txt", "alt", "title"):
            if k in j and isinstance(j[k], str):
                return j[k].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar-glob", default="/data/nitrot_ckpts/cc12m/cc12m-train-*.tar")
    ap.add_argument("--out-dir", default="/data/nitrot_ckpts/cc12m_split")
    ap.add_argument("--target-per-split", type=int, default=2500)
    ap.add_argument("--min-size", type=int, default=512,
                    help="Nitro-T's training filter threshold (min(H,W) >= this).")
    args = ap.parse_args()

    members_root = os.path.join(args.out_dir, "members")
    nonmembers_root = os.path.join(args.out_dir, "nonmembers")
    os.makedirs(members_root, exist_ok=True)
    os.makedirs(nonmembers_root, exist_ok=True)

    members_count = 0
    nonmembers_count = 0
    reason = defaultdict(int)

    tar_files = sorted(glob.glob(args.tar_glob))
    print(f"[info] {len(tar_files)} tar shards in glob")

    for tar_path in tar_files:
        print(f"[info] scanning {tar_path} ; members={members_count}, non={nonmembers_count}")
        for uid, sample in iter_wds_samples(tar_path):
            img_bytes = None
            for ext in IMG_EXTS:
                if ext in sample:
                    img_bytes = sample[ext]
                    break
            if img_bytes is None:
                reason["no_image"] += 1
                continue
            caption = extract_caption(sample)
            if not caption:
                reason["no_caption"] += 1
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                reason["bad_image"] += 1
                continue
            w, h = img.size
            if min(w, h) >= args.min_size:
                kind, count_ptr = "members", "m"
                if members_count >= args.target_per_split:
                    continue
                members_count += 1
                root = members_root
            else:
                kind, count_ptr = "nonmembers", "n"
                if nonmembers_count >= args.target_per_split:
                    continue
                nonmembers_count += 1
                root = nonmembers_root

            base = os.path.basename(uid)  # uid stripped from '<file>.ext'
            sub = base[:2] if len(base) >= 2 else "_"
            out_sub = os.path.join(root, sub)
            os.makedirs(out_sub, exist_ok=True)
            img.save(os.path.join(out_sub, f"{base}.jpg"), quality=92)
            with open(os.path.join(out_sub, f"{base}.txt"), "w") as fh:
                fh.write(caption)
            if members_count >= args.target_per_split and nonmembers_count >= args.target_per_split:
                break
        if members_count >= args.target_per_split and nonmembers_count >= args.target_per_split:
            break

    print(f"[done] members={members_count} non-members={nonmembers_count}")
    print(f"[skipped reasons] {dict(reason)}")


if __name__ == "__main__":
    main()
