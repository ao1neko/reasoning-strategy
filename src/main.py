import argparse
import torch
from distutils.util import strtobool
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Config
from transformers import BartForConditionalGeneration
from dentaku_tokenizer.tokenizer import T5DentakuTokenizer, BartDentakuTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np
import os
from typing import List, Dict
import re
from utils import read_jsonl_files, all_at_once_predict, step_by_step_predict, clean_text, clean_html_tags, eliminate_calculated_index, token_by_token_predict, token_by_token_eliminate_calculated_index
from models.modeling_shape_bart import SHAPEBartForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleDataset(Dataset):
    def __init__(self, inputs: Dict[str, torch.Tensor], labels: torch.Tensor, pad_token_id: int):
        self.inputs = inputs
        self.labels = labels
        self.labels[self.labels[:, :] == pad_token_id] = -100

    def __getitem__(self, idx: int):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def inputs_and_labels_to_dataset(tokenizer, inputs, labels, model_name: str = "t5"):
    if model_name == "t5":
        inputs = ["calculate: " + x for x in inputs]
    tokenized_inputs = tokenizer(
        inputs, padding=True, truncation=True, return_tensors='pt')
    tokenized_labels = tokenizer(
        labels, padding=True, truncation=True, return_tensors='pt')
    return SimpleDataset(tokenized_inputs, tokenized_labels["input_ids"], tokenizer.pad_token_id)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed = args.seed
    architecture_name = args.architecture_name
    train_dataset_names = args.train_dataset_names.split(",")
    valid_dataset_names = args.valid_dataset_names.split(",")
    test_dataset_names = args.test_dataset_names.split(",")
    load_model_dir = args.load_model_dir
    model_name = args.model_name
    train = strtobool(args.train)
    predict = strtobool(args.predict)
    pretrained_model = strtobool(args.pretrained_model)
    train_epochs = args.train_epochs
    eval_steps = args.eval_steps
    save_steps = args.save_steps
    output_dir = args.output_dir
    run_dir = args.run_dir
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    load_tokenizer = args.load_tokenizer
    log_level = "passive" if train else "critical"

    assert architecture_name in ["all_at_once", "step_by_step",
                                 "no_reasoning_step", "token_by_token"], "architecture name is incorrect"
    assert model_name in ["bart", "t5",
                          "shape_bart"], "model name is incorrect"

    train_inputs, train_labels = read_jsonl_files(train_dataset_names)
    valid_inputs, valid_labels = read_jsonl_files(valid_dataset_names)

    if model_name == "t5":
        if pretrained_model:
            model = T5ForConditionalGeneration.from_pretrained(
                load_model_dir).to(device)
        else:
            config = T5Config.from_pretrained(load_model_dir)
            model = T5ForConditionalGeneration(config=config).to(device)
        tokenizer = T5DentakuTokenizer.from_pretrained(load_tokenizer)

    elif model_name == "bart":
        model = BartForConditionalGeneration.from_pretrained(
            load_model_dir).to(device)
        tokenizer = BartDentakuTokenizer.from_pretrained(load_tokenizer)
        if load_model_dir == "facebook/bart-large":
            model.model.shared.weight.data[0] = model.model.shared.weight.data[0] + torch.randn(
                1024).to(device)
    elif model_name == "shape_bart":
        model = SHAPEBartForConditionalGeneration.from_pretrained(
            load_model_dir).to(device)
        tokenizer = BartDentakuTokenizer.from_pretrained(load_tokenizer)
        if load_model_dir == "facebook/bart-large":
            model.model.shared.weight.data[0] = model.model.shared.weight.data[0] + torch.randn(
                1024).to(device)

    train_dataset = inputs_and_labels_to_dataset(
        tokenizer, train_inputs, train_labels, model_name)
    valid_dataset = inputs_and_labels_to_dataset(
        tokenizer, valid_inputs, valid_labels, model_name)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        run_name=run_dir,
        load_best_model_at_end=True,
        predict_with_generate=True,
        disable_tqdm=predict,
        log_level=log_level,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    if train:
        trainer.train()
    if predict:
        test_inputs, test_labels = read_jsonl_files(test_dataset_names)
        acc_num = 0
        inference_acc_num = 0
        data_size = len(test_labels)
        label_length_list = []
        predict_length_list = []

        if architecture_name in ["no_reasoning_step", "all_at_once"]:
            test_dataset = inputs_and_labels_to_dataset(
                tokenizer, test_inputs, test_labels, model_name)

            predict_id_list, label_id_list, metrics = trainer.predict(
                test_dataset=test_dataset, max_length=500)
            acc_num, inference_acc_num, label_length_list, predict_length_list = all_at_once_predict(
                test_dataset.inputs.input_ids, predict_id_list, label_id_list, tokenizer, output_dir, model_name)

        elif architecture_name == "step_by_step":
            MAX_STEPS = 300 if model_name == "bart" else 100

            for i in range(MAX_STEPS):
                test_dataset = inputs_and_labels_to_dataset(
                    tokenizer, test_inputs, test_labels, model_name)

                predict_id_list, label_id_list, metrics = trainer.predict(
                    test_dataset=test_dataset, max_length=500)
                each_acc_num, each_inference_acc_num, calculated_list, predict_text_list,  label_length_list, predict_length_list = step_by_step_predict(
                    test_dataset.inputs.input_ids, predict_id_list, label_id_list, tokenizer, output_dir, i, model_name, label_length_list, predict_length_list)
                acc_num += each_acc_num
                inference_acc_num += each_inference_acc_num

                predict_text_list = [clean_html_tags(
                    x) for x in predict_text_list]
                test_inputs = list(
                    map(lambda x, y: x + " " + y, test_inputs, predict_text_list))
                test_inputs, test_labels = eliminate_calculated_index(
                    calculated_list, test_inputs, test_labels)
                if len(test_inputs) == 0:
                    break

        elif architecture_name == "token_by_token":
            MAX_STEPS = 1000 if model_name == "bart" else 500

            for i in range(MAX_STEPS):
                test_dataset = inputs_and_labels_to_dataset(
                    tokenizer, test_inputs, test_labels, model_name)

                predict_id_list, label_id_list, metrics = trainer.predict(
                    test_dataset=test_dataset, max_length=5)
                each_acc_num, each_inference_acc_num, calculated_list, predict_text_list, label_length_list, predict_length_list = token_by_token_predict(
                    test_dataset.inputs.input_ids, predict_id_list, label_id_list, tokenizer, output_dir, i, model_name, label_length_list, predict_length_list)
                acc_num += each_acc_num
                inference_acc_num += each_inference_acc_num

                predict_text_list = [clean_text(x) for x in predict_text_list]
                test_inputs = [x + y if x[-1].isdecimal() and y.isdecimal()
                               else x + " " + y for x, y in zip(test_inputs, predict_text_list)]
                test_inputs, test_labels = token_by_token_eliminate_calculated_index(
                    calculated_list, test_inputs, test_labels)
                if len(test_inputs) == 0:
                    break

        print(f"acc: {acc_num/data_size}")
        print(f"inference acc {inference_acc_num/data_size}")
        # print(f"label_length_list:{label_length_list}")
        # print(f"predict_length_list:{predict_length_list}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--architecture_name', default="no_reasoning_step")
    parser.add_argument('--train_dataset_names', default="train1,train2")
    parser.add_argument('--valid_dataset_names', default="valid1,valid2")
    parser.add_argument('--test_dataset_names', default="test1,valid2")
    parser.add_argument('--model_name', default="bart")
    parser.add_argument('--load_model_dir', default="facebook/bart-base")
    parser.add_argument('--load_tokenizer', default="facebook/bart-base")
    parser.add_argument('--pretrained_model', default='true')
    parser.add_argument('--train', default='false')
    parser.add_argument('--predict', default='false')
    parser.add_argument('--train_epochs', default=100, type=int)
    parser.add_argument('--eval_steps', default=100, type=int)
    parser.add_argument('--save_steps', default=10000, type=int)
    parser.add_argument('--output_dir', default="save/test")
    parser.add_argument('--run_dir', default="save/test")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    args = parser.parse_args()
    main(args)
