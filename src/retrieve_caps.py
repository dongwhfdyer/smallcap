import itertools
import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision.transforms import functional as F


def load_huggingface_model(model_class, model_name, local_file, **kwargs):
    if os.path.exists(local_file):
        print(f"Loading {model_class} from local file {local_file}")
        model = model_class.from_pretrained(local_file)
    else:
        print(f"Loading {model_class} from huggingface model {model_name}")
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(local_file)
        Path(local_file).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saved {model_class} to local file {local_file}")
    if kwargs.get("return_tokenizer", None) == True:
        return model.tokenizer
    elif kwargs.get("return_feature_extractor", None) == True:
        return model.feature_extractor
    elif kwargs.get("return_vision_model", None) == True:
        return model.vision_model
    return model


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_coco_data(coco_data_path):
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


def filter_captions(data):
    decoder_name = '.cache/gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    bs = 512

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    # find the maximum length of the captions
    # max_len = 0
    # for cap in caps:
    #     input_ids = tokenizer.encode(cap, return_tensors='pt')
    #     if max_len < len(input_ids[0]):
    #         max_len = len(input_ids[0])

    encodings = []
    # for idx in range(0, len(data), bs):
    #     encodings += tokenizer.batch_encode_plus(caps[idx:idx + bs], return_tensors='np', padding= 'max_length', max_length=56)['input_ids'].tolist()

    for idx in range(0, len(data), bs):
        # just using the `tokenizer.encode` method here
        sub_encodings = []
        for cap in caps[idx:idx + bs]:
            input_ids = tokenizer.encode(cap, return_tensors='np')
            sub_encodings.append(input_ids[0])
        encodings += sub_encodings

    # testttt = tokenizer.batch_encode_plus(caps[idx:idx + bs], return_tensors='np')
    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions


def encode_captions(captions, model, device):
    bs = 256
    encoded_captions = []

    for idx in tqdm(range(0, len(captions), bs)):
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx + bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())

    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions

def five_crop(image, ratio=0.6):
    w, h = image.size
    hw = (h * ratio, w * ratio)

    return F.five_crop(image, hw)

def nine_crop(image, ratio=0.4):
    w, h = image.size


    t = (0, int((0.5 - ratio / 2) * h), int((1.0 - ratio) * h))
    b = (int(ratio * h), int((0.5 + ratio / 2) * h), h)
    l = (0, int((0.5 - ratio / 2) * w), int((1.0 - ratio) * w))
    r = (int(ratio * w), int((0.5 + ratio / 2) * w), w)
    h, w = list(zip(t, b)), list(zip(l, r))

    images = []
    for s in itertools.product(h, w):
        h, w = s
        top, left = h[0], w[0]
        height, width = h[1] - h[0], w[1] - w[0]
        images.append(F.crop(image, top, left, height, width))

    return images




def encode_images(images, image_path, model, feature_extractor, device):
    image_ids = [i['image_id'] for i in images]

    bs = 64
    image_features = []

    for idx in tqdm(range(0, len(images), bs)):
        img = Image.open(os.path.join(image_path, i['file_name']))
        img_nine_crop = nine_crop(img)

        image_input = [feature_extractor()
                       for i in images[idx:idx + bs]]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)

    return image_ids, image_features


def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k)

    return index, I


def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in zip(nns_list):
            if xb_image_ids[nn[0]] == image_id:
                continue
            good_nns.append(captions[nn[0]])
            if len(good_nns) == 7:
                break
        assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions


def main():
    coco_data_path = 'data/dataset_coco.json'  # path to Karpathy splits downloaded from Kaggle
    image_path = 'data/images/'

    print('Loading data')
    images, captions = load_coco_data(coco_data_path)
    decoder_name = 'gpt2'
    # tokenizer = load_huggingface_model(AutoTokenizer, decoder_name, ".cache/gpt2_tokenizer.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, feature_extractor = clip.load("RN50x64", device=device)

    print('Filtering captions')
    xb_image_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, device)

    print('Encoding images')
    xq_image_ids, encoded_images = encode_images(images, image_path, clip_model, feature_extractor, device)

    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)

    print('Writing files')
    faiss.write_index(index, "datastore/coco_index")
    json.dump(captions, open('datastore/coco_index_captions.json', 'w'))

    json.dump(retrieved_caps, open('data/retrieved_caps_resnet50x64.json', 'w'))


if __name__ == '__main__':
    main()
