import argparse
from cmath import e, log
from operator import eq
from tkinter import N
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
import re


class Equation():
    def __init__(self, left_side=[], right_side=[], left_char_set=set(), right_char_set=set()):
        self.left_side = left_side
        self.right_side = right_side
        self.left_char_set = left_char_set
        self.right_char_set = right_char_set

    def calc_char_set(self):
        self.left_char_set = {x for x in self.left_side if type(x) is str}
        self.right_char_set = {x for x in self.right_side if type(x) is str}

    def calc_side_int_sum(self, calc_side="left"):
        if calc_side == "left":
            return sum([x for x in self.left_side if type(x) is int]) % 100
        elif calc_side == "right":
            return sum([x for x in self.right_side if type(x) is int]) % 100

    def to_str(self):
        left_side = ",".join([str(x) for x in self.left_side])
        right_side = ",".join([str(x) for x in self.right_side])
        return left_side + "=" + right_side

    def from_str(self, s: str):
        (left_side, right_side) = s.split("=")
        left_side = left_side.split(",")
        right_side = right_side.split(",")
        self.left_side = [self._convert_int(x) for x in left_side]
        self.right_side = [self._convert_int(x) for x in right_side]
        self.calc_char_set()

    def _is_int(self, s):
        s = str(s)
        return True if re.fullmatch('[0-9]+', s) else False

    def _convert_int(self, str):
        if self._is_int(str):
            return int(str)
        else:
            return str


class DataInstance():
    def __init__(self, useable_char_set: Set = set(string.ascii_uppercase), useable_int_set: Set = set(range(100))):
        self.inference_step = None
        self.equations = []
        self.shortest_inference_equations = []
        self.exhaustive_inference_equations = []
        self.backward_inference_equations = []
        self.question = None
        self.answer = None
        self.char_set = set()
        self.useable_char_set = useable_char_set
        self.useable_int_set = useable_int_set

    def to_str(self):
        equations_str = " , ".join([" + ".join([str(arg) for arg in x.left_side]) +
                                   " = " + " + ".join([str(arg) for arg in x.right_side]) for x in self.equations])
        shortest_inference_equations_str = " , ".join([" + ".join([str(arg) for arg in x.left_side]) + " = " + " + ".join(
            [str(arg) for arg in x.right_side]) for x in self.shortest_inference_equations])
        exhaustive_inference_equations_str = " , ".join([" + ".join([str(arg) for arg in x.left_side]) + " = " + " + ".join(
            [str(arg) for arg in x.right_side]) for x in self.exhaustive_inference_equations])
        backward_inference_equations_str = " , ".join([" + ".join([str(arg) for arg in x.left_side]) + " = " + " + ".join(
            [str(arg) for arg in x.right_side]) for x in self.backward_inference_equations])
        question_str = self.question + " ? "
        answer_str = str(self.answer)
        return (equations_str, question_str, answer_str, shortest_inference_equations_str, exhaustive_inference_equations_str, backward_inference_equations_str)

    def to_json(self):
        json_data = {
            "equations": [x.to_str() for x in self.equations],
            "question": self.question,
            "answer": self.answer,
            "inference_step": self.inference_step,
            "inference_equations": {
                "shortest_inference_equations": [x.to_str() for x in self.shortest_inference_equations],
                "exhaustive_inference_equations": [x.to_str() for x in self.exhaustive_inference_equations],
                "backward_inference_equations": [x.to_str() for x in self.backward_inference_equations],
            },
            "inform": {
                "char_set": list(self.char_set),
                "useable_char_set": list(self.useable_char_set),
            }
        }
        return json_data

    def from_json(self, json_data: str):
        json_dict = json.loads(json_data)
        self.inference_step = json_dict["inference_step"]
        self.equations = self._make_equations_from_str(json_dict["equations"])
        self.shortest_inference_equations = self._make_equations_from_str(
            json_dict["inference_equations"]["shortest_inference_equations"])
        self.exhaustive_inference_equations = self._make_equations_from_str(
            json_dict["inference_equations"]["exhaustive_inference_equations"])
        self.backward_inference_equations = self._make_equations_from_str(
            json_dict["inference_equations"]["backward_inference_equations"])
        self.question = json_dict["question"]
        self.answer = json_dict["answer"]
        self.char_set = set(json_dict["inform"]["char_set"])
        self.useable_char_set = set(json_dict["inform"]["useable_char_set"])

    def make_instance(self, inference_step: int, equation_num: int):
        self.inference_step = inference_step
        self._make_minimum_instance(inference_step)
        for _ in range(equation_num - inference_step):
            self._make_relate_equation()

        random.shuffle(self.equations)
        self._make_shortest_inference_equations()
        self._make_exhaustive_inference_equations()
        self._make_backward_inference_equations()

    def _make_equations_from_str(self, str_equations: List[Equation]):
        equations = []
        for str_equation in str_equations:
            equation = Equation()
            equation.from_str(str_equation)
            equations.append(equation)
        return equations

    # Make the smallest data instance that satisfies the specified inference step
    def _make_minimum_instance(self,
                               inference_step: int,):
        useable_char_set = self.useable_char_set.copy()
        self.question = random.choice(list(useable_char_set))
        self.answer = None
        must_use_char_set = set(self.question)

        useable_char_set = useable_char_set - must_use_char_set

        for index in range(inference_step):
            must_use_char_set_num = len(must_use_char_set)
            remain_equation_num = inference_step - index

            if remain_equation_num == must_use_char_set_num:
                equation_right_int_num = 1
                equation_right_char_num = 0
            elif remain_equation_num-1 < must_use_char_set_num+1:
                equation_right_int_num = 1
                equation_right_char_num = 1
            else:
                args_tyep = np.random.choice(
                    ["int_num_0_char_num_2", "int_num_1_char_num_1"], p=[0.5, 0.5])
                if args_tyep == "int_num_0_char_num_2":
                    equation_right_int_num = 0
                    equation_right_char_num = 2
                elif args_tyep == "int_num_1_char_num_1":
                    equation_right_int_num = 1
                    equation_right_char_num = 1

            left_char_set = set(must_use_char_set.pop())
            equation = self._make_equation(
                equation_left_char_set=left_char_set,
                equation_right_char_set=useable_char_set,
                equation_left_int_num=0,
                equation_left_char_num=1,
                equation_right_int_num=equation_right_int_num,
                equation_right_char_num=equation_right_char_num,
            )
            self.equations.append(equation)

            must_use_char_set = must_use_char_set | equation.right_char_set
            useable_char_set = useable_char_set - equation.right_char_set

    # Make a distractor
    def _make_relate_equation(self,):
        useable_char_set = self.useable_char_set.copy() - self.char_set
        if len(self.char_set) >= 2:
            args_tyep = np.random.choice(
                ["int_num_0_char_num_2", "int_num_1_char_num_1", "int_num_1_char_num_0"], p=[0.333, 0.333, 0.334])
        else:
            args_tyep = np.random.choice(
                ["int_num_1_char_num_1", "int_num_1_char_num_0"], p=[0.5, 0.5])

        if args_tyep == "int_num_0_char_num_2":
            equation_right_int_num = 0
            equation_right_char_num = 2
        elif args_tyep == "int_num_1_char_num_1":
            equation_right_int_num = 1
            equation_right_char_num = 1
        elif args_tyep == "int_num_1_char_num_0":
            equation_right_int_num = 1
            equation_right_char_num = 0

        related_equation = self._make_equation(
            equation_left_char_set=useable_char_set,
            equation_right_char_set=self.char_set,
            equation_left_int_num=0,
            equation_left_char_num=1,
            equation_right_int_num=equation_right_int_num,
            equation_right_char_num=equation_right_char_num,
        )
        self.equations.append(related_equation)

    def _make_equation(self,
                       equation_left_char_set: Set = set(
                           string.ascii_uppercase),
                       equation_right_char_set: Set = set(
                           string.ascii_uppercase),
                       equation_left_int_num: int = 0,
                       equation_left_char_num: int = 1,
                       equation_right_int_num: int = 1,
                       equation_right_char_num: int = 0):
        equation = Equation()
        equation_left_chars = random.choices(
            list(equation_left_char_set), k=equation_left_char_num)
        equation_right_chars = random.choices(
            list(equation_right_char_set), k=equation_right_char_num)
        equation.left_side = equation_left_chars + \
            random.choices(list(self.useable_int_set), k=equation_left_int_num)
        equation.right_side = equation_right_chars + \
            random.choices(list(self.useable_int_set),
                           k=equation_right_int_num)
        equation.calc_char_set()
        random.shuffle(equation.left_side)
        random.shuffle(equation.right_side)
        self.char_set = self.char_set | equation.left_char_set | equation.right_char_set

        return equation

    # For chaining strategy

    def _make_solved_char_dict(self):
        solved_char_dict = dict()
        for equation in self.equations:
            if len(equation.right_side) == 1 and len(equation.right_char_set) == 0:
                key = equation.left_side[0]
                value = equation.right_side[0]
                solved_char_dict[key] = value
        return solved_char_dict

    def _search_relate_equation(self, search_char: str, search_side: str = "left", equations="equations"):
        if equations == "equations":
            equations = self.equations
        elif equations == "shortest_inference_equations":
            equations = self.shortest_inference_equations
        elif equations == "exhaustive_inference_equations":
            equations = self.exhaustive_inference_equations
        elif equations == "backward_inference_equations":
            equations = self.backward_inference_equations

        if search_side == "left":
            for equation in equations:
                if search_char in equation.left_side:
                    return copy.deepcopy(equation)
        elif search_side == "right":
            for equation in equations:
                if search_char in equation.right_side:
                    return copy.deepcopy(equation)
        return None

    def _make_shortest_inference_equations(self):
        solved_char_dict = self._make_solved_char_dict()
        question_stack = []
        not_search_question_stack = []
        not_search_question_stack.append(self.question)

        while len(not_search_question_stack) != 0:
            search_char = not_search_question_stack.pop()
            equation = self._search_relate_equation(
                search_char=search_char, search_side="left")
            search_char_list = [
                x for x in equation.right_side if type(x) is str]

            question_stack.append(search_char)
            not_search_question_stack += search_char_list
        question_stack = list(set(question_stack) -
                              set(solved_char_dict.keys()))

        while len(question_stack) != 0:
            for equation in self.equations:
                calculate_flag = True
                if equation.left_side[0] in question_stack:
                    for search_char in equation.right_char_set:
                        if search_char not in solved_char_dict.keys():
                            calculate_flag = False
                            break
                    if calculate_flag == True:
                        self.shortest_inference_equations.append(
                            copy.deepcopy(equation))
                        search_char = equation.left_side[0]

                        while search_char in question_stack:
                            equation = copy.deepcopy(
                                self.shortest_inference_equations[-1])
                            if len(equation.right_char_set) == 0:
                                if len(equation.right_side) == 1:
                                    key = equation.left_side[0]
                                    value = equation.right_side[0]
                                    solved_char_dict[key] = value
                                    question_stack.remove(key)
                                else:
                                    equation.right_side = [
                                        equation.calc_side_int_sum(calc_side="right")]
                                    equation.calc_char_set()
                                    self.shortest_inference_equations.append(
                                        equation)
                            else:
                                temp_search_char = [
                                    x for x in equation.right_side if type(x) is str][0]
                                replace_index = equation.right_side.index(
                                    temp_search_char)
                                equation.right_side[replace_index] = solved_char_dict[temp_search_char]
                                equation.calc_char_set()
                                self.shortest_inference_equations.append(
                                    equation)
                        break
        if self.answer is None:
            self.answer = solved_char_dict[self.question]

    def _make_exhaustive_inference_equations(self):
        solved_char_dict = self._make_solved_char_dict()

        while self.question not in solved_char_dict.keys():
            for equation in self.equations:
                calculate_flag = True
                if equation.left_side[0] not in solved_char_dict.keys():
                    for search_char in equation.right_char_set:
                        if search_char not in solved_char_dict.keys():
                            calculate_flag = False
                            break
                    if calculate_flag == True:
                        self.exhaustive_inference_equations.append(
                            copy.deepcopy(equation))
                        search_char = equation.left_side[0]

                        while search_char not in solved_char_dict.keys():
                            equation = copy.deepcopy(
                                self.exhaustive_inference_equations[-1])
                            if len(equation.right_char_set) == 0:
                                if len(equation.right_side) == 1:
                                    key = equation.left_side[0]
                                    value = equation.right_side[0]
                                    solved_char_dict[key] = value
                                else:
                                    equation.right_side = [
                                        equation.calc_side_int_sum(calc_side="right")]
                                    equation.calc_char_set()
                                    self.exhaustive_inference_equations.append(
                                        equation)
                            else:
                                temp_search_char = [
                                    x for x in equation.right_side if type(x) is str][0]
                                replace_index = equation.right_side.index(
                                    temp_search_char)
                                equation.right_side[replace_index] = solved_char_dict[temp_search_char]
                                equation.calc_char_set()
                                self.exhaustive_inference_equations.append(
                                    equation)
                        break
        if self.answer is None:
            self.answer = solved_char_dict[self.question]

    def _make_backward_inference_equations(self):
        solved_char_dict = self._make_solved_char_dict()
        goal_equation = self._search_relate_equation(
            search_char=self.question, search_side="left")

        if self.question not in solved_char_dict.keys():
            self.backward_inference_equations.append(goal_equation)
        while self.question not in solved_char_dict.keys():
            backward_equation = copy.deepcopy(
                self.backward_inference_equations[-1])
            if len(backward_equation.right_char_set) == 0:
                if len(backward_equation.right_side) == 1:
                    search_char = backward_equation.left_side[0]
                    equation = self._search_relate_equation(
                        search_char=search_char, search_side="right", equations="backward_inference_equations")
                    replace_index = equation.right_side.index(search_char)
                    equation.right_side[replace_index] = solved_char_dict[search_char]
                    equation.calc_char_set()

                    self.backward_inference_equations.append(equation)
                else:
                    equation = backward_equation
                    equation.right_side = [
                        equation.calc_side_int_sum(calc_side="right")]
                    equation.calc_char_set()
                    key = equation.left_side[0]
                    value = equation.right_side[0]
                    solved_char_dict[key] = value
                    self.backward_inference_equations.append(equation)
            else:
                search_char = [
                    x for x in backward_equation.right_side if type(x) is str][0]
                if search_char in solved_char_dict.keys():
                    equation = backward_equation
                    replace_index = equation.right_side.index(search_char)
                    equation.right_side[replace_index] = solved_char_dict[search_char]
                    equation.calc_char_set()
                    self.backward_inference_equations.append(equation)
                else:
                    equation = self._search_relate_equation(
                        search_char=search_char, search_side="left")
                    self.backward_inference_equations.append(equation)
        if self.answer is None:
            self.answer = solved_char_dict[self.question]


class NaturalLanguage():
    def __init__(self, *args, **kwargs):
        self.variable_to_name_dict = {
            "A": "Alyssa",
            "B": "Benny",
            "C": "Christina",
            "D": "Dan",
            "E": "Ella",
            "F": "Fred",
            "G": "Greg",
            "H": "Hannah",
            "I": "Irwin",
            "J": "Joan",
            "K": "Keith",
            "L": "Lisa",
            "M": "Mike",
            "N": "Nicole",
            "O": "Oscar",
            "P": "Pamela",
            "Q": "Quincy",
            "R": "Ron",
            "S": "Sara",
            "T": "Tim",
            "U": "Ulysses",
            "V": "Victor",
            "W": "Wendy",
            "X": "Xaver",
            "Y": "Yvonne",
            "Z": "Zachary",
        }

    def variable_to_name(self, variable: str):
        if variable in self.variable_to_name_dict.keys():
            return self.variable_to_name_dict[variable]
        else:
            return variable


class NaturalLanguageEquation(Equation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_natural_language_str(self):
        NL = NaturalLanguage()
        int_list = []
        char_list = []

        for x in self.right_side:
            if self._is_int(x):
                int_list.append(x)
            else:
                char_list.append(x)

        if len(char_list) == 0 and len(int_list) == 1:
            return f"{NL.variable_to_name(self.left_side[0])} had {int_list[0]} ENTITY ."
        elif len(char_list) == 1 and len(int_list) == 1:
            return f"{NL.variable_to_name(self.left_side[0])} had {int_list[0]} more ENTITY than {NL.variable_to_name(char_list[0])} had ."
        elif len(int_list) == 2:
            return f"{NL.variable_to_name(self.left_side[0])} had {int_list[0]} + {int_list[1]} ENTITY ."
        elif len(char_list) == 2:
            return f"{NL.variable_to_name(self.left_side[0])} had as many ENTITY as {NL.variable_to_name(char_list[0])} and {NL.variable_to_name(char_list[1])} ."
        else:
            print(f"incorrect equation: {self.right_side}")
            exit()


class NaturalLanguageDataInstance(DataInstance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.natural_language_equations = []
        self.natural_language_shortest_inference_equations = []
        self.natural_language_exhaustive_inference_equations = []
        self.natural_language_backward_inference_equations = []
        self.natural_language_question = None
        self.entities = ["apples", "books", "cups", "dolls", "flowers"]

    def to_str(self):
        entity = random.choice(self.entities)
        NL = NaturalLanguage()
        equations_str = " ".join([re.sub(r"ENTITY", entity, x.to_natural_language_str(
        )) for x in self.natural_language_equations])
        shortest_inference_equations_str = " ".join([re.sub(r"ENTITY", entity, x.to_natural_language_str(
        )) for x in self.natural_language_shortest_inference_equations])
        exhaustive_inference_equations_str = " ".join([re.sub(r"ENTITY", entity, x.to_natural_language_str(
        )) for x in self.natural_language_exhaustive_inference_equations])
        backward_inference_equations_str = " ".join([re.sub(r"ENTITY", entity, x.to_natural_language_str(
        )) for x in self.natural_language_backward_inference_equations])
        question_str = f"How many {entity} does {NL.variable_to_name(self.question)} + have ?"
        answer_str = str(self.answer)
        return (equations_str, question_str, answer_str, shortest_inference_equations_str, exhaustive_inference_equations_str, backward_inference_equations_str)

    def from_json(self, json_data: str):
        super().from_json(json_data)
        self.natural_language_equations = [NaturalLanguageEquation(
            x.left_side, x.right_side, x.left_char_set, x.right_char_set) for x in self.equations]
        self.natural_language_shortest_inference_equations = [NaturalLanguageEquation(
            x.left_side, x.right_side, x.left_char_set, x.right_char_set) for x in self.shortest_inference_equations]
        self.natural_language_exhaustive_inference_equations = [NaturalLanguageEquation(
            x.left_side, x.right_side, x.left_char_set, x.right_char_set) for x in self.exhaustive_inference_equations]
        self.natural_language_backward_inference_equations = [NaturalLanguageEquation(
            x.left_side, x.right_side, x.left_char_set, x.right_char_set) for x in self.backward_inference_equations]
        self.natural_language_question = self.question
