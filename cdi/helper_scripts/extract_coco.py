#!/usr/bin/env python3
"""Extract COCO zip files."""
import zipfile
import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
coco_dir = repo_root / "data" / "coco"
os.chdir(coco_dir)

for zf in ['annotations_trainval2014.zip', 'val2014.zip', 'train2014.zip']:
    if os.path.exists(zf):
        print(f'Extracting {zf}...')
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall('.')
        print(f'{zf} done')
    else:
        print(f'{zf} not found')

print('All done!')
