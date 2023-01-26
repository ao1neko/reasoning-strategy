import argparse
from cmath import e, log
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path
from tqdm import tqdm
import re
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict, Set, Tuple
import string
import copy
import json
import math
from numerical_data_class import DataInstance, NaturalLanguageDataInstance


def convert_numerical_datasets_no_reasoning_step(data_instance: DataInstance) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    label = "answer : " + answer_str
    yield (input, label)


def convert_numerical_datasets_all_at_once_dot(data_instance: DataInstance) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    label = " , ".join(["." for _ in range(data_instance.inference_step)])
    if len(label) == 0:
        label += " answer : " + answer_str
    else:
        label += " , answer : " + answer_str
    yield (input, label)


def convert_numerical_datasets_step_by_step_dot(data_instance: DataInstance, test: bool = False) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str + "</s>"
    inference_equations = ["." for _ in range(data_instance.inference_step)]

    if not test:
        inference_equations = inference_equations + ["answer : " + answer_str]
        yield (input, inference_equations[0])

        for i in range(1, len(inference_equations)):
            if i == 1:
                input += " " + inference_equations[i-1]
            else:
                input += " , " + inference_equations[i-1]
            yield (input, inference_equations[i])
    else:
        input = equations_str + " </s> " + question_str + "</s>"
        label = " , ".join(inference_equations)
        if len(label) == 0:
            label += " answer : " + answer_str
        else:
            label += " , answer : " + answer_str
        yield (input, label)


def convert_numerical_datasets_all_at_once(data_instance: DataInstance, inference_type="shortest") -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    if inference_type == "shortest":
        label = shortest_inference_equations_str
    elif inference_type == "exhaustive":
        label = exhaustive_inference_equations_str
    elif inference_type == "backward":
        label = backward_inference_equations_str

    if len(label) == 0:
        label += " answer : " + answer_str
    else:
        label += " , answer : " + answer_str
    yield (input, label)


def convert_numerical_datasets_step_by_step(data_instance: DataInstance, inference_type="shortest", test: bool = False) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str + "</s>"
    if inference_type == "shortest":
        inference_equations = data_instance.shortest_inference_equations
    elif inference_type == "exhaustive":
        inference_equations = data_instance.exhaustive_inference_equations
    elif inference_type == "backward":
        inference_equations = data_instance.backward_inference_equations
    inference_equations = [" + ".join([str(arg) for arg in x.left_side]) + " = " + " + ".join(
        [str(arg) for arg in x.right_side]) for x in inference_equations]

    if not test:
        inference_equations = [
            x + " ," for x in inference_equations] + ["answer : " + answer_str]
        yield (input, inference_equations[0])

        for i in range(1, len(inference_equations)):
            input += " " + inference_equations[i-1]
            yield (input, inference_equations[i])
    else:
        if inference_type == "shortest":
            label = shortest_inference_equations_str
        elif inference_type == "exhaustive":
            label = exhaustive_inference_equations_str
        elif inference_type == "backward":
            label = backward_inference_equations_str

        if len(label) == 0:
            label += " answer : " + answer_str
        else:
            label += " , answer : " + answer_str
        yield (input, label)


def convert_numerical_datasets_token_by_token(data_instance: DataInstance, inference_type="shortest", test: bool = False) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str + "</s>"
    if inference_type == "shortest":
        inference_equation_str = shortest_inference_equations_str
    elif inference_type == "exhaustive":
        inference_equation_str = exhaustive_inference_equations_str
    elif inference_type == "backward":
        inference_equation_str = backward_inference_equations_str
    inference_equation_str = inference_equation_str.replace(" ", "")

    if not test:
        for i in range(0, len(inference_equation_str)):
            yield (input, inference_equation_str[i])
            if input[-1].isdecimal() and inference_equation_str[i].isdecimal():
                input += inference_equation_str[i]
            else:
                input += " " + inference_equation_str[i]
        append_inference = ["answer", ":"]+list(answer_str)+["</s>"] if len(
            inference_equation_str) == 0 else [",", "answer", ":"]+list(answer_str)+["</s>"]
        for text in append_inference:
            yield (input, text)
            if input[-1].isdecimal() and text.isdecimal():
                input += text
            else:
                input += " " + text

    else:
        if inference_type == "shortest":
            label = shortest_inference_equations_str
        elif inference_type == "exhaustive":
            label = exhaustive_inference_equations_str
        elif inference_type == "backward":
            label = backward_inference_equations_str

        if len(label) == 0:
            label += " answer : " + answer_str
        else:
            label += " , answer : " + answer_str
        yield (input, label)


def convert_NL_numerical_datasets_no_reasoning_step(data_instance: NaturalLanguageDataInstance,) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    label = "answer : " + answer_str
    yield (input, label)


def convert_NL_numerical_datasets_all_at_once(data_instance: NaturalLanguageDataInstance, inference_type="shortest") -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str
    if inference_type == "shortest":
        label = shortest_inference_equations_str
    elif inference_type == "exhaustive":
        label = exhaustive_inference_equations_str
    elif inference_type == "backward":
        label = backward_inference_equations_str

    label += " answer : " + answer_str
    yield (input, label)


def convert_NL_numerical_datasets_step_by_step(data_instance: NaturalLanguageDataInstance, inference_type="shortest", test: bool = False) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str + "</s>"
    if inference_type == "shortest":
        inference_equations_str = shortest_inference_equations_str
    elif inference_type == "exhaustive":
        inference_equations_str = exhaustive_inference_equations_str
    elif inference_type == "backward":
        inference_equations_str = backward_inference_equations_str

    if not test:
        inference_equations_str_list = [
            x + "." for x in inference_equations_str.split(".")][:-1] + ["answer : " + answer_str]
        yield (input, inference_equations_str_list[0])

        for i in range(1, len(inference_equations_str_list)):
            input += " " + inference_equations_str_list[i-1]
            yield (input, inference_equations_str_list[i])
    else:
        if inference_type == "shortest":
            label = shortest_inference_equations_str
        elif inference_type == "exhaustive":
            label = exhaustive_inference_equations_str
        elif inference_type == "backward":
            label = backward_inference_equations_str
        label += " answer : " + answer_str
        yield (input, label)


def convert_NL_numerical_datasets_token_by_token(data_instance: NaturalLanguageDataInstance, inference_type="shortest", test: bool = False) -> Tuple[List[str], List[str]]:
    (equations_str, question_str, answer_str, shortest_inference_equations_str,
     exhaustive_inference_equations_str, backward_inference_equations_str) = data_instance.to_str()
    input = equations_str + " </s> " + question_str + "</s>"
    if inference_type == "shortest":
        inference_equation_str = shortest_inference_equations_str
    elif inference_type == "exhaustive":
        inference_equation_str = exhaustive_inference_equations_str
    elif inference_type == "backward":
        inference_equation_str = backward_inference_equations_str

    if not test:
        inference_equation_str += f"answer : {answer_str} </s>"
        inference_equation_str = re.sub(
            r"([0-9)])([0-9])", r"\1 \2", inference_equation_str)
        inference_equation_str_list = inference_equation_str.split(" ")

        for i in range(0, len(inference_equation_str_list)):
            yield (input, inference_equation_str_list[i])
            if input[-1].isdecimal() and inference_equation_str_list[i].isdecimal():
                input += inference_equation_str_list[i]
            else:
                input += " " + inference_equation_str_list[i]

    else:
        if inference_type == "shortest":
            label = shortest_inference_equations_str
        elif inference_type == "exhaustive":
            label = exhaustive_inference_equations_str
        elif inference_type == "backward":
            label = backward_inference_equations_str
        label += " answer : " + answer_str
        yield (input, label)


def convert_numerical_datasets(output_json_file: str, generator):
    with open(output_json_file, 'a') as output_file_object:
        for input, label in generator:
            converted_json_data = {
                "input": input,
                "label": label
            }
            output_file_object.write(
                json.dumps((converted_json_data)))
            output_file_object.write("\n")


def delete_file_name(file_parent: str, file_stem: str, file_tail: str):
    delete_file_path = file_parent / (file_stem + "_" + file_tail + ".jsonl")
    if (os.path.isfile(delete_file_path)):
        os.remove(delete_file_path)


def make_output_file_name(input_file_parent: str, input_file_stem: str, file_tail: str):
    return input_file_parent / (input_file_stem + "_" + file_tail + ".jsonl")


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    input_json_file = Path(args.input_file)
    input_json_file_stem = input_json_file.stem
    input_json_file_NL_stem = input_json_file.stem + "_NL"
    input_json_file_parent = input_json_file.parent

    method = args.method
    test = True if method == "test" else False

    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "no_reasoning_step")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "all_at_once_shortest")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "all_at_once_exhaustive")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "all_at_once_backward")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "step_by_step_shortest")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "step_by_step_exhaustive")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "step_by_step_backward")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "token_by_token_shortest"),
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "token_by_token_exhaustive")
    delete_file_name(input_json_file_parent,
                     input_json_file_stem, "token_by_token_backward")

    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "no_reasoning_step")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "all_at_once_shortest")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "all_at_once_exhaustive")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "all_at_once_backward")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "step_by_step_shortest")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "step_by_step_exhaustive")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "step_by_step_backward")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "token_by_token_shortest"),
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "token_by_token_exhaustive")
    delete_file_name(input_json_file_parent,
                     input_json_file_NL_stem, "token_by_token_backward")

    with open(input_json_file, 'r') as input_file_object:
        for json_data in input_file_object:
            if json_data != "\n":
                data_instance = DataInstance()
                data_instance.from_json(json_data)

                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem,
                                           "no_reasoning_step"), convert_numerical_datasets_no_reasoning_step(data_instance))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem,
                                           "all_at_once_shortest"), convert_numerical_datasets_all_at_once(data_instance, inference_type="shortest"))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "all_at_once_exhaustive"),
                                           convert_numerical_datasets_all_at_once(data_instance, inference_type="exhaustive"))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem,
                                           "all_at_once_backward"), convert_numerical_datasets_all_at_once(data_instance, inference_type="backward"))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "step_by_step_shortest"),
                                           convert_numerical_datasets_step_by_step(data_instance, inference_type="shortest", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "step_by_step_exhaustive"),
                                           convert_numerical_datasets_step_by_step(data_instance, inference_type="exhaustive", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "step_by_step_backward"),
                                           convert_numerical_datasets_step_by_step(data_instance, inference_type="backward", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "token_by_token_shortest"),
                                           convert_numerical_datasets_token_by_token(data_instance, inference_type="shortest", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "token_by_token_exhaustive"),
                                           convert_numerical_datasets_token_by_token(data_instance, inference_type="exhaustive", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_stem, "token_by_token_backward"),
                                           convert_numerical_datasets_token_by_token(data_instance, inference_type="backward", test=test))

                data_instance = NaturalLanguageDataInstance()
                data_instance.from_json(json_data)
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem,
                                           "no_reasoning_step"), convert_NL_numerical_datasets_no_reasoning_step(data_instance))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem,
                                           "all_at_once_shortest"), convert_NL_numerical_datasets_all_at_once(data_instance, inference_type="shortest"))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "all_at_once_exhaustive"),
                                           convert_NL_numerical_datasets_all_at_once(data_instance, inference_type="exhaustive"))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem,
                                           "all_at_once_backward"), convert_NL_numerical_datasets_all_at_once(data_instance, inference_type="backward"))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "step_by_step_shortest"),
                                           convert_NL_numerical_datasets_step_by_step(data_instance, inference_type="shortest", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "step_by_step_exhaustive"),
                                           convert_NL_numerical_datasets_step_by_step(data_instance, inference_type="exhaustive", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "step_by_step_backward"),
                                           convert_NL_numerical_datasets_step_by_step(data_instance, inference_type="backward", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "token_by_token_shortest"),
                                           convert_NL_numerical_datasets_token_by_token(data_instance, inference_type="shortest", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "token_by_token_exhaustive"),
                                           convert_NL_numerical_datasets_token_by_token(data_instance, inference_type="exhaustive", test=test))
                convert_numerical_datasets(make_output_file_name(input_json_file_parent, input_json_file_NL_stem, "token_by_token_backward"),
                                           convert_NL_numerical_datasets_token_by_token(data_instance, inference_type="backward", test=test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--input_file', default="./data/depth_1_num_1_train.jsonal", type=str)
    parser.add_argument('--method', default="train", type=str)
    parser.add_argument('--seed', default=10, type=int)
    args = parser.parse_args()
    main(args)
