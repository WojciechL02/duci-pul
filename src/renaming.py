import os
import re

# Directory containing your .npz files
DIR = "../data/"  # change if needed

# Pattern components
mapping_rules = [
    # (pattern, replacement)
    (r"real_imagenet_val",   "real_nonmem_suspect_5k"),
    (r"real_imagenet_train", "real_mem_suspect_5k"),
    (r"tp_s0\.5_imagenet_train", "ae_mem_suspect_5k"),
    (r"fp_s0\.5_imagenet_train", "ae_nonmem_suspect_5k"),
    (r"tp_s0\.5_imagenet_val", "generated_nonmem_generated_from-mem_5k"),
    (r"fp_s0\.5_imagenet_val", "generated_nonmem_added_from-nonmem_5k"),
]

for fname in os.listdir(DIR):
    if not fname.endswith(".npz"):
        continue

    new_name = fname
    for pat, rep in mapping_rules:
        new_name = re.sub(pat, rep, new_name)

    if new_name != fname:
        src = os.path.join(DIR, fname)
        dst = os.path.join(DIR, new_name)
        print(f"Renaming:\n  {fname}\n→ {new_name}\n")
        os.rename(src, dst)
