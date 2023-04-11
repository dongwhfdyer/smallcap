import argparse
import json
import os
from pathlib import Path
import shutil
import h5py
import numpy as np
import torch

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer

import sys

from utils import load_huggingface_model

sys.path.append('.')
from torch.utils.data import Dataset

LENGTH_LIMIT = 25


def collate_tokens(batch):
    captions, input_ids, attention_mask, lengths = [], [], [], []
    for cap, tok in batch:
        assert tok["input_ids"].shape == tok["attention_mask"].shape
        captions.append(cap)

        l = tok["input_ids"].shape[1]
        if l < LENGTH_LIMIT:
            input_ids.append(tok["input_ids"])
            attention_mask.append(tok["attention_mask"])
            lengths.append(l)
        else:
            input_ids.append(tok["input_ids"][:, :LENGTH_LIMIT])
            attention_mask.append(tok["attention_mask"][:, :LENGTH_LIMIT])
            lengths.append(LENGTH_LIMIT)

    max_len = max(lengths)
    input_pad, atten_pad = [], []
    for i in range(len(input_ids)):
        l = input_ids[i].shape[1]
        if l < max_len:
            p = torch.zeros(size=(1, max_len - l), dtype=input_ids[i].dtype)
            input_pad.append(torch.cat([input_ids[i], p], dim=1))

            p = torch.zeros(size=(1, max_len - l), dtype=attention_mask[i].dtype)
            atten_pad.append(torch.cat([attention_mask[i], p], dim=1))
        else:
            input_pad.append(input_ids[i])
            atten_pad.append(attention_mask[i])

    input_pad = torch.cat(input_pad)
    atten_pad = torch.cat(atten_pad)
    assert input_pad.shape[1] <= LENGTH_LIMIT
    assert atten_pad.shape[1] <= LENGTH_LIMIT
    assert input_pad.shape == atten_pad.shape

    tokens = {"input_ids": input_pad, "attention_mask": atten_pad}

    return captions, tokens


class CaptionDB(LightningModule):
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        LOCAL_CLIP_FILE = ".cache/clip-vit-base-patch32"
        self.clip = load_huggingface_model(CLIPModel, CLIP_MODEL_NAME, LOCAL_CLIP_FILE)
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None
        # captions: (batch_size, max_len)
        # tokens: input_ids: (batch_size, max_len), attention_mask: (batch_size, max_len)
        captions, tokens = batch
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        # self.clip.text_model(**tokens): last_hidden_state (bs, max_len, 512), pooler_output (bs, 512)
        # So, It's extracting the pooler_output
        features = self.clip.text_model(**tokens)[1]
        keys = self.clip.text_projection(features)
        keys = keys / keys.norm(dim=-1, keepdim=True)

        features = features.detach().cpu().numpy()
        keys = keys.detach().cpu().numpy()

        # captions(list): bs
        # features(ndarray): (bs, embed_dim)
        # keys(ndarray): (bs, text_projection_dim)
        with h5py.File(self.save_dir / "caption_db.hdf5", "a") as f:
            g = f.create_group(str(batch_idx))
            g.create_dataset("keys", data=keys, compression="gzip")
            g.create_dataset("features", data=features, compression="gzip")
            g.create_dataset("captions", data=captions, compression="gzip")


class cocoCaptions(Dataset):
    def __init__(self, COCO_DATA_PATH):
        super().__init__()
        CLIPPROCESSOR_NAME = "openai/clip-vit-base-patch32"
        LOCAL_CLIPPROCESSOR_FILE = ".cache/clip-vit-base-patch32"
        LOCAL_CAPS_FILE = ".cache/captions.txt"
        self.tokenizer = load_huggingface_model(CLIPProcessor, CLIPPROCESSOR_NAME, LOCAL_CLIPPROCESSOR_FILE, return_tokenizer=True)
        self.caps = self.read_caps_cache_or_process(LOCAL_CAPS_FILE, COCO_DATA_PATH)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.caps[index], padding=True, return_tensors="pt")

        return self.caps[index], tokens

    def read_caps_cache_or_process(self, local_caps_file, COCO_DATA_PATH):
        if Path(local_caps_file).exists():
            with open(local_caps_file, 'r') as f:
                filtered_captions = f.read().splitlines()
        else:
            _, captions = self.load_coco_data(COCO_DATA_PATH)
            decoder_name = '.cache/gpt2'
            tokenizer = AutoTokenizer.from_pretrained(decoder_name)
            bs = 512

            image_ids = [d['image_id'] for d in captions]
            caps = [d['caption'] for d in captions]

            encodings = []

            for idx in range(0, len(captions), bs):
                sub_encodings = []
                for cap in caps[idx:idx + bs]:
                    input_ids = tokenizer.encode(cap, return_tensors='np')
                    sub_encodings.append(input_ids[0])
                encodings += sub_encodings

            filtered_image_ids, filtered_captions = [], []

            assert len(image_ids) == len(caps) and len(caps) == len(encodings)
            for image_id, cap, encoding in zip(image_ids, caps, encodings):
                if len(encoding) <= 25:
                    filtered_image_ids.append(image_id)
                    filtered_captions.append(cap)

            with open(local_caps_file, 'w') as f:
                # save caps
                f.write('\n'.join(filtered_captions))

        return filtered_captions

    def load_coco_data(self, coco_data_path):
        """We load in all images and only the train captions."""

        annotations = json.load(open(coco_data_path))['images']
        images = []
        captions = []
        for item in annotations:
            if item['split'] == 'restval':
                item['split'] = 'train'
            if item['split'] == 'train':
                for sentence in item['sentences']:
                    captions.append({'image_id': item['cocoid'], 'caption': ' '.join(sentence['tokens'])})
            images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})

        return images, captions

    def __len__(self):
        return len(self.caps)


def build_caption_db(args):
    dset = cocoCaptions(args.ann_file)
    dloader = DataLoader(
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_tokens
    )
    cap_db = CaptionDB(args.save_dir)

    trainer = Trainer(
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir

    )
    trainer.test(cap_db, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode captions')
    parser.add_argument('--device', type=int, default=[0, 1, 2, 3], nargs='+', help='GPU device(s) to use')
    parser.add_argument('--exp_name', type=str, default='captions_db')
    parser.add_argument('--ann_file', type=str, default='data/dataset_coco.json')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    setattr(args, "save_dir", Path("experiments") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    with open((args.save_dir / 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(args)

    seed_everything(1, workers=True)

    build_caption_db(args)
