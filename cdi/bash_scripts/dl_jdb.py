#!/usr/bin/env python3
"""Download JourneyDB shard 000 of train + valid + the two anno jsonl tarballs."""
import os
from huggingface_hub import hf_hub_download

DEST = "/data/nitrot_ckpts/jdb"
TOKEN = os.environ.get("HF_TOKEN")
os.makedirs(DEST, exist_ok=True)

for sub, fn in [
    ("data/valid", "valid_anno_repath.jsonl.tgz"),
    ("data/train", "train_anno_realease_repath.jsonl.tgz"),
    ("data/valid/imgs", "000.tgz"),
    ("data/train/imgs", "000.tgz"),
]:
    p = hf_hub_download(
        repo_id="JourneyDB/JourneyDB",
        repo_type="dataset",
        filename=f"{sub}/{fn}",
        local_dir=DEST,
        token=TOKEN,
    )
    print(f"got {p} ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)

print("JDB_DONE", flush=True)
