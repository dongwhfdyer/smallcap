import subprocess
import time

import pandas as pd
import numpy as np
import os
import argparse
import torch
import torchvision
from transformers.models.auto.configuration_auto import AutoConfig
from transformers import AutoTokenizer, CLIPFeatureExtractor, AutoModel, AutoModelForCausalLM, TrainingArguments, TrainerState, TrainerControl
from transformers import Seq2SeqTrainer, default_data_collator, Seq2SeqTrainingArguments

from src.vision_encoder_decoder import SmallCap, SmallCapConfig
from src.gpt2 import ThisGPT2Config, ThisGPT2LMHeadModel
from src.utils import *

# for attention with 28M params, we devide the attention dimensions by 1
# for attention with 14M params, we devide the attention dimensions by 2, etc.
PARAMS2REDUCE_FACTOR = {28: 1, 14: 2, 7: 4, 3.5: 8, 1.75: 16}
PAD_TOKEN = '!'
EOS_TOKEN = '.'
CAPTION_LENGTH = 25

from transformers import TrainerCallback


class InferenceCallback(TrainerCallback):
    # def on_epoch_end(self, args, state, control, **kwargs):
    #     saved_one = state.global_step
    #     print("saved one: ", saved_one)
    #     subprocess.run(["bash", "infer_eval.sh", str(saved_one)])

    # def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     saved_one = state.global_step
    #     print("saved one: ", saved_one)
    #     subprocess.run(["bash", "infer_eval.sh", str(saved_one)])

    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     pass
    pass


# class smallcapTrainer(Seq2SeqTrainer):


def get_model_and_auxiliaries(args):
    # register model types
    AutoConfig.register("this_gpt2", ThisGPT2Config)
    AutoModel.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoModelForCausalLM.register(ThisGPT2Config, ThisGPT2LMHeadModel)
    AutoConfig.register("smallcap", SmallCapConfig)
    AutoModel.register(SmallCapConfig, SmallCap)

    # create and configure model
    cross_attention_reduce_factor = PARAMS2REDUCE_FACTOR[args.attention_size]

    feature_extractor = load_huggingface_model(CLIPFeatureExtractor, args.encoder_name, ".cache/CLIPFeatureExtractor.pt")
    tokenizer = load_huggingface_model(AutoTokenizer, args.decoder_name, ".cache/AutoTokenizer.pt")
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN

    model = SmallCap.from_encoder_decoder_pretrained(args.encoder_name,
                                                     args.decoder_name,
                                                     cross_attention_reduce_factor=cross_attention_reduce_factor)
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.decoder_start_token_id = None
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not args.disable_rag:
        model.config.k = args.k
        model.config.retrieval_encoder = args.retrieval_encoder
        model.config.max_length = args.k * CAPTION_LENGTH + CAPTION_LENGTH + 18  # there are 18 tokens in the long prefix template. kuhn: I changed `4 * CAPTION_LENGTH` to `args.k * CAPTION_LENGTH `

    else:
        model.config.max_length = CAPTION_LENGTH + 4  # there are 4 tokens in the short prefix template
    model.config.rag = not args.disable_rag

    # freeze parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    if not args.train_decoder:
        for name, param in model.decoder.named_parameters():
            if 'crossattention' not in name:
                param.requires_grad = False

    # count trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Training a model with {} trainable parameters.'.format(num_trainable_params))

    return model, tokenizer, feature_extractor


def get_data(tokenizer, max_length, args):
    data = load_data_for_training_v1(args.annotations_path, args.captions_path)  # todo: whether to use the original retrieve caps or not
    # data = load_data_for_training_v2(args.annotations_path, args.retrieved_caps_path)
    train_df = pd.DataFrame(data['train'])

    train_dataset = TrainDataset_v2(
        df=train_df,
        features_path=os.path.join(args.features_dir, 'coco_cdn.hdf5'),
        tokenizer=tokenizer,
        rag=not args.disable_rag,
        template_path=args.template_path,
        k=args.k,
        max_target_length=max_length)

    return train_dataset


# class myTrainer(Seq2SeqTrainer):
#     def eva


def main(args):
    model, tokenizer, feature_extractor = get_model_and_auxiliaries(args)
    train_dataset = get_data(tokenizer, model.config.max_length, args)

    model_type = 'norag' if args.disable_rag else 'rag'
    if args.resume_cpt is None:
        output_dir = 'exp_{}'.format(time.strftime("%m%d-%H%M", time.localtime()))
        output_dir = os.path.join(args.experiments_dir, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = args.resume_cpt

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    training_args = Seq2SeqTrainingArguments(
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_steps,
        learning_rate=args.lr,
        fp16=True,
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=1,
        # save_total_limit=args.n_epochs,
        logging_strategy="epoch",
        output_dir=output_dir,
        overwrite_output_dir=True,
        # resume_from_checkpoint=args.resume_cpt,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=feature_extractor,
    )
    trainer.add_callback(InferenceCallback)
    trainer.train()  # todo: whether to resume from checkpoint
    # trainer.train(resume_from_checkpoint=args.resume_cpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument("--features_dir", type=str, default="features/", help="Directory where cached input image features are stored")
    parser.add_argument("--annotations_path", type=str, default="data/dataset_coco.json", help="JSON file with annotations in Karpathy splits")
    parser.add_argument("--experiments_dir", type=str, default="experiments/", help="Directory where trained models will be saved")

    parser.add_argument("--encoder_name", type=str, default="openai/clip-vit-base-patch32", help="Encoder name as found of HuggingFace or stored locally")
    parser.add_argument("--decoder_name", type=str, default="gpt2", help="Decoder name as found of HuggingFace or stored locally")
    parser.add_argument("--attention_size", type=float, default=7, help="Number of parameters in the cross attention {28, 14, 7, 3.5, 1.75}")
    parser.add_argument("--train_decoder", action="store_true", default=False, help="Whether to train the decoder in addition to the attention")

    parser.add_argument("--disable_rag", action="store_true", default=False, help="Disable retrieval augmentation")
    parser.add_argument("--k", type=int, default=2, help="Number of retrieved captions to use in prefix")  # todo: it should be set to 9 when training with block-splited retrieved captions
    parser.add_argument("--retrieval_encoder", type=str, default="RN50x64", help="Visual encoder used for retieving captions")
    parser.add_argument("--captions_path", type=str, default="data/retrieved_caps_resnet50x64.json", help="JSON file with retrieved captions")
    parser.add_argument("--template_path", type=str, default="src/template.txt", help="TXT file with template")

    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")  # todo: when k=9, batch_size should be smaller. Because
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient accumulation steps")

    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")

    # parser.add_argument("--retrieved_caps_path", type=str, default="data/coco2017_crop_caps.hdf5")  # todo: decide what retrieved caps to use. The name is important
    parser.add_argument("--retrieved_caps_path", type=str, default="experiments/coco2017all_crop/coco2017_crop_long_caps.hdf5")  # todo: decide what retrieved caps to use. The name is important
    parser.add_argument("--resume_cpt", type=str, default=None, help="Path to checkpoint to resume training from or leave None to train from scratch")
    args = parser.parse_args()

    main(args)
