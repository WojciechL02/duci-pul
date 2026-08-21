#!/usr/bin/env python3
"""Download N shards of CC12M WDS to a target dir."""
import os
import sys
from huggingface_hub import hf_hub_download


def main():
    n_shards = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    dest = sys.argv[2] if len(sys.argv) > 2 else "/data/nitrot_ckpts/cc12m"
    os.makedirs(dest, exist_ok=True)
    for i in range(n_shards):
        p = hf_hub_download(
            repo_id="pixparse/cc12m-wds",
            repo_type="dataset",
            filename=f"cc12m-train-{i:>04}.tar",
            local_dir=dest,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"shard {i} -> {p} ({os.path.getsize(p)} B)", flush=True)
    print("CC12M_DONE", flush=True)


if __name__ == "__main__":
    main()
