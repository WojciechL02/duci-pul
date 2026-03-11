import os
import re

# Directory containing your .npz files
DIR = "../data/"
save_dir = "../data/"

# Pattern components
mapping_rules = [
    # (pattern, replacement)
    (r"real_imagenet_val", "real_nonmem_suspect_5k"),
    (r"real_imagenet_train", "real_mem_suspect_5k"),
    (rf"tp_.*_train", "ae_mem_suspect_5k"),
    (rf"fp_.*_train", "ae_nonmem_suspect_5k"),
    (rf"tp_.*_val", "generated_nonmem_generated_from-mem_5k"),
    (rf"fp_.*_val", "generated_nonmem_added_from-nonmem_5k"),
]

for fname in os.listdir(DIR):
    if not fname.endswith(".npz"):
        continue

    new_name = fname
    for pat, rep in mapping_rules:
        new_name = re.sub(pat, rep, new_name)

    src = os.path.join(DIR, fname)
    dst = os.path.join(save_dir, new_name)
    print(f"Renaming:\n  {fname}\n→ {new_name}\n")
    os.rename(src, dst)
