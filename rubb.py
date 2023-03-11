from pathlib import Path

import h5py
from tqdm import tqdm

coco_all_path = Path("data/coco_all.hdf5")
coco_nine_caption_only_path = Path("data/coco_nine_caption_only.hdf5")

with h5py.File(coco_all_path, "r") as f:
    with h5py.File(coco_nine_caption_only_path, "w") as f2:
        for gg in tqdm(f.keys()):
            g_name = gg
            g = f2.create_dataset(g_name, data=f[gg]["nine"]["texts"])
