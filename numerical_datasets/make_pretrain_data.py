import argparse
from cmath import e, log
from operator import mod
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


def make_reference_data(json_file, useable_char_set: Set = set(string.ascii_uppercase)):
    min_num = 0
    max_num = 99
    for i in range(min_num, max_num+1):
        question = random.choice(list(useable_char_set))

        equation = Equation()
        equation.left_side = [question]
        equation.right_side = [i]
        equation.calc_char_set()

        data_instance = DataInstance(useable_char_set=useable_char_set)
        data_instance.inference_step = 0
        data_instance.equations.append(equation)
        data_instance.question = question
        data_instance.answer = i
        data_instance.char_set = set(question)

        json_file.write(json.dumps((data_instance.to_json())))
        json_file.write("\n")


def make_add_data(json_file, useable_char_set: Set = set(string.ascii_uppercase)):
    min_num = 0
    max_num = 99
    mod_num = 100
    for i in range(min_num, max_num+1):
        for j in range(min_num, max_num+1):
            question = random.choice(list(useable_char_set))

            equation = Equation()
            equation.left_side = [question]
            equation.right_side = [i, j]
            equation.calc_char_set()

            inference_equation = Equation()
            inference_equation.left_side = [question]
            inference_equation.right_side = [(i + j) % mod_num]
            inference_equation.calc_char_set()

            data_instance = DataInstance(useable_char_set=useable_char_set)
            data_instance.inference_step = 0
            data_instance.equations.append(equation)
            data_instance.shortest_inference_equations.append(
                inference_equation)
            data_instance.exhaustive_inference_equations.append(
                inference_equation)
            data_instance.backward_inference_equations.append(
                inference_equation)
            data_instance.question = question
            data_instance.answer = (i + j) % mod_num
            data_instance.char_set = set(question)

            json_file.write(json.dumps((data_instance.to_json())))
            json_file.write("\n")


def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    output_dir = Path("./data/pretrain/")
    output_file = output_dir / Path("pretrain.jsonl")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(output_file, 'w') as output_json_file:
        make_add_data(json_file=output_json_file)
        make_reference_data(json_file=output_json_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', default=10, type=int)
    args = parser.parse_args()
    main(args)
