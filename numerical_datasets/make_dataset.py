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
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict, Set
import string
import copy
import json

from numerical_data_class import DataInstance, Equation


def make_json_number_set(seed: int = 10):
    number_list = set(range(100))
    train_number_set = set(random.sample(number_list, 80))
    number_list = number_list - train_number_set

    valid_number_set = set(random.sample(number_list, 10))
    test_number_set = number_list - valid_number_set
    json_number_set = {
        "train_number_set": train_number_set,
        "valid_number_set": valid_number_set,
        "test_number_set": test_number_set
    }

    return json_number_set


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    train_data_size = args.train_data_size
    valid_data_size = args.valid_data_size
    test_data_size = args.test_data_size
    output_dir = Path(args.output_dir)

    inference_step = args.inference_step
    equation_num = args.equation_num
    output_dir = output_dir / \
        Path(f"depth_{inference_step}_distractor_{equation_num-inference_step}")

    json_number_set = make_json_number_set()
    number_set_list = [json_number_set["train_number_set"],
                       json_number_set["valid_number_set"], json_number_set["test_number_set"]]
    data_size_list = [train_data_size, valid_data_size, test_data_size]
    method_list = ["train", "valid", "test"]

    for number_set, data_size, method in zip(number_set_list, data_size_list, method_list):
        maked_data_size = 0
        equations_stack = []
        output_file = output_dir / Path(f"{method}.jsonl")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(Path(output_file), 'w') as json_file:
            while data_size > maked_data_size:
                data_instance = DataInstance(useable_char_set=set(
                    string.ascii_uppercase), useable_int_set=number_set)
                data_instance.make_instance(
                    inference_step=inference_step, equation_num=equation_num)
                equations_str, question_str, answer_str, shortest_inference_equations_str, exhaustive_inference_equations_str, backward_inference_equations_str = data_instance.to_str()
                if equations_str not in equations_stack:
                    equations_stack.append(equations_str)
                    maked_data_size += 1
                    json_file.write(json.dumps((data_instance.to_json())))
                    json_file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_data_size', default=100, type=int)
    parser.add_argument('--valid_data_size', default=100, type=int)
    parser.add_argument('--test_data_size', default=100, type=int)
    parser.add_argument('--inference_step', default=1, type=int)
    parser.add_argument('--equation_num', default=1, type=int)
    parser.add_argument('--output_dir', default="./data", type=str)
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    main(args)
