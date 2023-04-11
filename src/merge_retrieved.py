from pathlib import Path

import h5py
from tqdm import tqdm


workspace = Path("experiments/coco2017all_crop/")
outputpath = workspace / "coco2017_crop_caps.hdf5"

concat_num = 4

files = [workspace / "txt_ctx_{:d}.hdf5".format(i) for i in range(concat_num)]

# read hdf5 and save to the outputpath
with h5py.File(outputpath, "w") as f:
    for file in files:
        with h5py.File(file, "r") as f1:
            for gg in tqdm(f1.keys()):
                g_name = gg
                for gg2 in f1[gg].keys():
                    g = f.create_group(g_name + "/" + gg2)
                    g.create_dataset("features", data=f1[gg][gg2]["features"])
                    g.create_dataset("texts", data=f1[gg][gg2]["texts"])
