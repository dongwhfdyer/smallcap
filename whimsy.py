# ---------kkuhn-block------------------------------ # coco_nine_caption_only.hdf5
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
# ---------kkuhn-block------------------------------


# ---------kkuhn-block------------------------------ # final_outputs.hdf5
from pathlib import Path
import h5py
import numpy as np

feature_outputs_PATH = Path(r"features/final_outputs.hdf5")

# read hdf5

with h5py.File(feature_outputs_PATH, "r") as f:
    for gg in f.keys():
        print(gg)
        print(f[gg].keys())
        value = f[gg]['final_out']
        # convert to numpy array
        value_ = np.array(value)
        print("--------------------------------------------------")
# ---------kkuhn-block------------------------------
