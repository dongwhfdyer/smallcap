from pathlib import Path

from torch.utils.data import Dataset
from PIL import Image
import torch
import json
import h5py
import bisect

CAPTION_LENGTH = 25
SIMPLE_PREFIX = "This image shows "

import os


def load_huggingface_model(model_class, model_name, local_file, return_any=None, *model_args, **model_kwargs):
    if return_any is None:
        return_any = {}
    if os.path.exists(local_file):
        print(f"Loading {model_class} from local file {local_file}")
        model = model_class.from_pretrained(local_file, *model_args, **model_kwargs)
    else:
        print(f"Loading {model_class} from huggingface model {model_name}")
        model = model_class.from_pretrained(model_name, *model_args, **model_kwargs)
        model.save_pretrained(local_file)
        print(f"Saved {model_class} to local file {local_file}")
    if return_any.get("return_tokenizer", None) == True:
        return model.tokenizer
    elif return_any.get("return_feature_extractor", None) == True:
        return model.feature_extractor
    elif return_any.get("return_vision_model", None) == True:
        return model.vision_model
    return model


def prep_strings(text, tokenizer, template=None, retrieved_caps=None, k=None, is_test=False, max_length=None):
    if is_test:
        padding = False
        truncation = False
    else:
        padding = True
        truncation = True

    if retrieved_caps is not None:  # It's the part where retrieve_caps are used
        infix = '\n\n'.join(retrieved_caps[:k]) + '.'
        prefix = template.replace('||', infix)
    else:
        prefix = SIMPLE_PREFIX

    prefix_ids = tokenizer.encode(prefix)
    len_prefix = len(prefix_ids)

    text_ids = tokenizer.encode(text)
    if truncation:
        text_ids = text_ids[:CAPTION_LENGTH]
    input_ids = prefix_ids + text_ids if not is_test else prefix_ids

    # we ignore the prefix (minus one as the first subtoken in the prefix is not predicted)
    label_ids = [-100] * (len_prefix - 1) + text_ids + [tokenizer.eos_token_id]
    if padding:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        label_ids += [-100] * (max_length - len(label_ids))
    # here len(input_ids) == len(label_ids)

    if is_test:
        return input_ids
    else:
        return input_ids, label_ids


def postprocess_preds(pred, tokenizer):
    pred = pred.split(SIMPLE_PREFIX)[-1]
    pred = pred.replace(tokenizer.pad_token, '')
    if pred.endswith(tokenizer.eos_token):
        pred = pred[:-1]
    return pred


class TrainDataset(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_target_length=150):
        self.df = df
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.features = h5py.File(features_path, 'r')

        if rag:
            self.template = open(template_path).read().strip() + ' '
            assert k is not None
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag:
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)
        # load precomputed features
        encoder_outputs = self.features[self.df['cocoid'][idx]][()]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs),
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


class TrainDataset_v2(Dataset):
    def __init__(self, df, features_path, tokenizer, rag=False, template_path=None, k=None, max_target_length=150):
        self.df = df
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.features = h5py.File(features_path, 'r')

        if rag:
            self.template = open(template_path).read().strip() + ' '
            assert k is not None
            self.k = k
        self.rag = rag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df['text'][idx]
        if self.rag:
            caps = self.df['caps'][idx]
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, template=self.template,
                                                     retrieved_caps=caps, k=self.k, max_length=self.max_target_length)
        else:
            decoder_input_ids, labels = prep_strings(text, self.tokenizer, max_length=self.max_target_length)
        # here encoder_outputs' shape is [100,1024] which is extracted from CDN. The original encoder_outputs' shape is [50, 768]
        encoder_outputs = self.features[self.df['cocoid'][idx].zfill(12)]['final_out'][()].squeeze()  # I truncate the shape to [100, 768] from [100, 1024]
        # encoder_outputs = self.features[self.df['cocoid'][idx].zfill(12)]['final_out'][()].squeeze()[:, :768] # I truncate the shape to [100, 768] from [100, 1024]
        encoding = {"encoder_outputs": torch.tensor(encoder_outputs),
                    "decoder_input_ids": torch.tensor(decoder_input_ids),
                    "labels": torch.tensor(labels)}

        return encoding


def load_data_for_training_v1(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']
    if caps_path is not None:
        retrieved_caps = json.load(open(caps_path))
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if caps_path is not None:
            caps = retrieved_caps[str(item['cocoid'])]
        else:
            caps = None
        samples = []
        for sentence in item['sentences']:
            samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens'])})
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'] += samples
        elif item['split'] == 'val':
            data['val'] += samples
    return data


def load_data_for_training_v2(annot_path, caps_path):
    annotations = json.load(open(annot_path))['images']

    retrieved_caps_handler = h5py.File(caps_path, 'r')

    data = {'train': [], 'val': []}
    # get the caps_path's name
    caps_name = Path(caps_path).stem

    if not os.path.exists(f'data/train_{caps_name}.json'):
        print("not found retrieved caption json, collecting data...")
        for item in annotations:
            file_name = item['filename'].split('_')[-1][:-4]
            caps = retrieved_caps_handler[file_name]['nine']['texts'][()]
            caps = [str(cap[0]).replace('b', '').replace("'", '') for cap in caps]
            samples = []
            for sentence in item['sentences']:
                samples.append({'file_name': file_name, 'cocoid': str(item['cocoid']), 'caps': caps, 'text': ' '.join(sentence['tokens'])})
            if item['split'] == 'train' or item['split'] == 'restval':
                data['train'] += samples
            elif item['split'] == 'val':
                data['val'] += samples

        # put data to json
        with open(f'data/train_{caps_name}.json', 'w') as f:
            json.dump(data, f)

        retrieved_caps_handler.close()

        return data
    else:
        print("found retrieved caption json, loading data...")
        with open(f'data/train_{caps_name}.json', 'r') as f:
            data = json.load(f)

        retrieved_caps_handler.close()

        return data


def load_data_for_inference(dataset_name, retrieved_source_name):
    # read datasets_info.yaml file
    yaml_path = 'datasets_info.yaml'
    with open(yaml_path, 'r') as f:
        datasets_info = yaml.load(f, Loader=yaml.FullLoader)

    def build_coco(annot_path, caps_path):
        annotations = json.load(open(annot_path))['images']
        if caps_path is not None:
            retrieved_caps = json.load(open(caps_path))
        data = {'test': [], 'val': []}

        for item in annotations:
            file_name = item['filename'].split('_')[-1]
            if caps_path is not None:
                caps = retrieved_caps[str(item['cocoid'])]
            else:
                caps = None
            image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
            if item['split'] == 'test':
                data['test'].append(image)
            elif item['split'] == 'val':
                data['val'].append(image)

        return data

    def build_flickr30k(annot_path, caps_path):
        annotations = json.load(open(annot_path))['images']
        if caps_path is not None:
            retrieved_caps = json.load(open(caps_path))



    def build_msr_vtt(annot_path, caps_path):
        pass

    def build_vizwiz(annot_path, caps_path):
        pass

    if dataset_name == 'coco':
        annot_path = datasets_info['coco']['annotations_path']
        caps_path = datasets_info['coco']['retrieved_caps_path']
        return build_coco(annot_path, caps_path)

    if dataset_name == 'flickr30k':
        annot_path = datasets_info['flickr30k']['annotations_path']
        caps_path = datasets_info['flickr30k']['retrieved_caps_path'][retrieved_source_name]
        return build_flickr30k(annot_path, caps_path)

    if dataset_name == 'msr_vtt':
        annot_path = datasets_info['msr_vtt']['annotations_path']
        caps_path = datasets_info['msr_vtt']['retrieved_caps_path'][retrieved_source_name]
        return build_msr_vtt(annot_path, caps_path)

    if dataset_name == 'vizwiz':
        annot_path = datasets_info['vizwiz']['annotations_path']
        caps_path = datasets_info['vizwiz']['retrieved_caps_path'][retrieved_source_name]
        return build_vizwiz(annot_path, caps_path)

    if dataset_name == 'nocaps':
        raise NotImplementedError


def load_data_for_inference_v2(annot_path, caps_path=None):
    annotations = json.load(open(annot_path))['images']

    retrieved_caps_handler = h5py.File(caps_path, 'r')

    data = {'test': [], 'val': []}

    caps_name = Path(caps_path).stem

    if not os.path.exists(f'data/test_{caps_name}.json'):
        for item in annotations:
            file_name = item['filename'].split('_')[-1][:-4]
            caps = retrieved_caps_handler[file_name]['nine']['texts'][()]
            caps = [str(cap[0]).replace('b', '').replace("'", '') for cap in caps]
            image = {'file_name': file_name, 'caps': caps, 'image_id': str(item['cocoid'])}
            if item['split'] == 'test':
                data['test'].append(image)
            elif item['split'] == 'val':
                data['val'].append(image)

        # put data to json
        with open(f'data/test_{caps_name}.json', 'w') as f:
            json.dump(data, f)

        retrieved_caps_handler.close()

    else:
        with open(f'data/test_{caps_name}.json', 'r') as f:
            data = json.load(f)

        retrieved_caps_handler.close()

        return data
