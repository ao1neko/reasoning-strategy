from pathlib import Path
from tqdm import tqdm
import os
from typing import List, Dict, Tuple
import re
import json


def read_jsonl_file(input_file: Path):
    with open(input_file, 'r') as input_json_file:
        for json_data in input_json_file:
            if json_data != "\n":
                json_data = json.loads(json_data)
                yield (json_data["input"], json_data["label"])


def read_jsonl_files(input_file_list: List[Path]):
    inputs = []
    labels = []
    for input_file in input_file_list:
        for (input, label) in read_jsonl_file(input_file):
            inputs.append(input)
            labels.append(label)
    return (inputs, labels)


def print_format(file_object, input_text, label_text, predict_text):
    file_object.write(f"input:{input_text}\n")
    file_object.write(f"label:{label_text}\n")
    file_object.write(f"predict:{predict_text}\n")


def retrieve_last_string(str: str) -> str:
    match = re.search(r".*answer :\s*(.+?)</s>", str)
    try:
        return match.group(1).replace(' ', '')
    except:
        return None


def clean_ids(ids, pad_id):
    cleaned_ids = [x for x in ids if x != pad_id]
    return cleaned_ids


def clean_html_tags(text: str) -> str:
    cleaned_text = re.sub(r"<pad>", "", text)
    cleaned_text = re.sub(r"</s>", "", cleaned_text)
    cleaned_text = re.sub(r"<s>", "", cleaned_text)
    return cleaned_text


def clean_text(text: str) -> str:
    cleaned_text = re.sub(" ", "", (text))
    cleaned_text = clean_html_tags(cleaned_text)
    return cleaned_text


def retrieve_inference(text):
    #text = text.replace(' ', '')
    match = re.search(r"</s>.*?</s>(.+?)answer", text)
    try:
        return match.group(1)
    except:
        return None


def eliminate_calculated_index(calculated_list: List[int], test_inputs: List[str], test_labels: List[str]) -> Tuple[List[str], List[str], List[str]]:
    eliminated_test_inputs = []
    eliminated_test_labels = []

    for i, (calculated_index, input, label) in enumerate(zip(calculated_list, test_inputs, test_labels)):
        if calculated_index is None:
            eliminated_test_inputs.append(input)
            eliminated_test_labels.append(label)
    return (eliminated_test_inputs, eliminated_test_labels)


def check_step_by_step_skip_inference_err(input_text, label_inference):
    input_inference, _, predict_inference, _ = input_text.split("</s>")
    input_inference = re.sub(r"calculate:", "", input_inference)
    input_inference = re.sub(r" ", "", input_inference)
    predict_inference = re.sub(r" ", "", predict_inference)
    label_inference = re.sub(r" ", "", label_inference)
    input_inference_list = input_inference.split(",")
    predict_inference_list = predict_inference.split(",")

    input_dict = {}
    for input in input_inference_list:
        if len(input.split("=")) != 2:
            continue
        left_arg, right_arg = input.split("=")
        if not right_arg.isdecimal():
            input_dict[left_arg] = right_arg

    first_step = True
    for predict in predict_inference_list:
        if len(predict.split("=")) != 2:
            continue
        left_arg, right_arg = predict.split("=")
        if first_step:
            first_step = False
            if right_arg != input_dict[left_arg]:
                return True
        if right_arg.isdecimal():
            first_step = True
    return False


def token_by_token_eliminate_calculated_index(last_token_list: List[int], test_inputs: List[str], test_labels: List[str]) -> Tuple[List[str], List[str], List[str]]:
    eliminated_test_inputs = []
    eliminated_test_labels = []

    for last_token_index, input, label in zip(last_token_list, test_inputs, test_labels):
        if not last_token_index:
            eliminated_test_inputs.append(input)
            eliminated_test_labels.append(label)
    return (eliminated_test_inputs, eliminated_test_labels)


def multi_decode(tokenizer, ids_list):
    decoded_list = []
    for ids in ids_list:
        decoded_text = tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decoded_list.append(decoded_text)
    return decoded_list


def make_variable_dict(inference_list):
    variable_dict = {}
    for inference in inference_list:
        if len(inference.split("=")) != 2:
            continue

        left_arg, right_arg = inference.split("=")
        if right_arg.isdecimal():
            variable_dict[left_arg] = right_arg

    return variable_dict


# predict
def all_at_once_predict(input_id_list, predict_id_list, label_id_list, tokenizer, output_dir, model_name) -> int:
    acc_num = 0
    inference_acc_num = 0
    analyze_file_path = Path(output_dir) / Path('analyze.txt')
    err_file_path = Path(output_dir) / Path('analyze_err.txt')
    inference_err_file_path = Path(
        output_dir) / Path('analyze_inference_err.txt')
    del_analyze_flag = True
    del_analyze_err_flag = True
    del_analyze_inference_err_flag = True
    label_length_list = []
    predict_length_list = []

    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err, open(inference_err_file_path, "w") as f_inference_err:
        for input_ids, predict_ids, label_ids in zip(input_id_list, predict_id_list, label_id_list):
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            if model_name == "bart":
                input_text, label_text, predict_text = multi_decode(
                    tokenizer, [input_ids, label_ids[1:], predict_ids[2:]])
            elif model_name == "t5":
                input_text, label_text, predict_text = multi_decode(
                    tokenizer, [input_ids, label_ids, predict_ids[1:]])

            predict = retrieve_last_string(predict_text)
            label = retrieve_last_string(label_text)

            # calc inference length
            cleaned_label_text = clean_text(label_text)
            cleaned_predict_text = clean_text(predict_text)
            label_length_list.append(len(cleaned_label_text))
            predict_length_list.append(len(cleaned_predict_text))

            if predict is not None and (predict == label):
                acc_num += 1
                if cleaned_label_text == cleaned_predict_text:
                    inference_acc_num += 1
                    del_analyze_flag = False
                    print_format(f, input_text, label_text, predict_text)
                else:
                    del_analyze_inference_err_flag = False
                    print_format(f_inference_err, input_text,
                                 label_text, predict_text)
            else:
                del_analyze_err_flag = False
                print_format(f_err, input_text, label_text, predict_text)

    if del_analyze_flag:
        os.remove(analyze_file_path)
    if del_analyze_err_flag:
        os.remove(err_file_path)
    if del_analyze_inference_err_flag:
        os.remove(inference_err_file_path)
    return acc_num, inference_acc_num, label_length_list, predict_length_list


def step_by_step_predict(input_id_list, predict_id_list, label_id_list, tokenizer, output_dir, step_index, model_name, label_length_list, predict_length_list):
    acc_num = 0
    inference_acc_num = 0
    calculated_list = []
    predict_text_list = []

    analyze_file_path = Path(output_dir) / Path(f'analyze_{step_index}.txt')
    err_file_path = Path(output_dir) / Path(f'analyze_err_{step_index}.txt')
    inference_err_file_path = Path(
        output_dir) / Path(f'analyze_inference_err_{step_index}.txt')
    over_file_path = Path(
        output_dir) / Path(f'analyze_over_err.txt')

    del_analyze_flag = True
    del_analyze_err_flag = True
    del_analyze_inference_err_flag = True

    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err, open(inference_err_file_path, "w") as f_inference_err:
        for input_ids, predict_ids, label_ids in zip(input_id_list, predict_id_list, label_id_list):
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            if model_name == "bart":
                input_text, label_text, predict_text = multi_decode(
                    tokenizer, [input_ids, label_ids[1:], predict_ids[2:]])
            elif model_name == "t5":
                input_text, label_text, predict_text = multi_decode(
                    tokenizer, [input_ids, label_ids, predict_ids[1:]])
            predict_text_list.append(predict_text)

            predict = retrieve_last_string(predict_text)
            calculated_list.append(predict)
            label = retrieve_last_string(label_text)
            predict_inference = input_text.split("</s>")[2]
            label_inference = label_text.split("answer")[0]

            # calc inference length
            if predict is not None:
                #cleaned_predict_text = clean_text(predict_inference+","+ predict_text)
                cleaned_predict_text = clean_text(
                    predict_inference + predict_text)
                cleaned_label_text = clean_text(label_text)
                label_length_list.append(len(cleaned_label_text))
                predict_length_list.append(len(cleaned_predict_text))

            if predict is not None and (predict == label):
                acc_num += 1
                if clean_text(predict_inference) == clean_text(label_inference):
                    inference_acc_num += 1
                    del_analyze_flag = False
                    print_format(f, input_text, label_text, predict_text)
                else:
                    del_analyze_inference_err_flag = False
                    print_format(f_inference_err, input_text,
                                 label_text, predict_text)
            elif predict is not None:
                del_analyze_err_flag = False
                print_format(f_err, input_text, label_text, predict_text)
    if step_index == 99:
        with open(over_file_path, "w") as f_over_err:
            print_format(f_over_err, input_text, label_text, predict_text)
    if del_analyze_flag:
        os.remove(analyze_file_path)
    if del_analyze_err_flag:
        os.remove(err_file_path)
    if del_analyze_inference_err_flag:
        os.remove(inference_err_file_path)

    return acc_num, inference_acc_num, calculated_list, predict_text_list, label_length_list, predict_length_list


def token_by_token_predict(input_id_list, predict_id_list, label_id_list, tokenizer, output_dir, step_index, model_name, label_length_list, predict_length_list):
    acc_num = 0
    inference_acc_num = 0
    calculated_list = []
    predict_text_list = []
    analyze_file_path = Path(output_dir) / Path(f'analyze_{step_index}.txt')
    err_file_path = Path(output_dir) / Path(f'analyze_err_{step_index}.txt')
    inference_err_file_path = Path(
        output_dir) / Path(f'analyze_inference_err_{step_index}.txt')
    over_file_path = Path(
        output_dir) / Path(f'analyze_over_err.txt')

    del_analyze_flag = True
    del_analyze_err_flag = True
    del_analyze_inference_err_flag = True
    with open(analyze_file_path, 'w') as f, open(err_file_path, 'w') as f_err, open(inference_err_file_path, "w") as f_inference_err:
        for input_ids, predict_ids, label_ids in zip(input_id_list, predict_id_list, label_id_list):
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            if model_name == "bart":
                input_text, label_text, predict_text = multi_decode(
                    tokenizer, [input_ids, label_ids[1:], predict_ids[2:]])
            elif model_name == "t5":
                input_text, label_text, predict_text = multi_decode(
                    tokenizer, [input_ids, label_ids, predict_ids[1:]])
            predict_text_list.append(predict_text)

            bool_last_token = predict_text.startswith("</s>")

            predict = retrieve_last_string(input_text)
            calculated_list.append(bool_last_token)
            label = retrieve_last_string(label_text)

            predict_inference = input_text.split("</s>")[2]
            if bool_last_token:
                cleaned_predict_text = clean_text(predict_inference)
                cleaned_label_text = clean_text(label_text)
                label_length_list.append(len(cleaned_label_text))
                predict_length_list.append(len(cleaned_predict_text))

            if bool_last_token and (predict == label):
                acc_num += 1
                predict_inference_str = retrieve_inference(input_text)
                label_inference_str = retrieve_inference("</s></s>"+label_text)

                if predict_inference_str == label_inference_str:
                    del_analyze_flag = False
                    inference_acc_num += 1
                    print_format(f, input_text, label_text, predict_text)
                else:
                    del_analyze_inference_err_flag = False
                    print_format(f_inference_err, input_text,
                                 label_text, predict_text)
            elif bool_last_token:
                del_analyze_err_flag = False
                print_format(f_err, input_text, label_text, predict_text)
    if step_index == 499:
        with open(over_file_path, "w") as f_over_err:
            print_format(f_over_err, input_text, label_text, predict_text)

    if del_analyze_flag:
        os.remove(analyze_file_path)
    if del_analyze_err_flag:
        os.remove(err_file_path)
    if del_analyze_inference_err_flag:
        os.remove(inference_err_file_path)

    return acc_num, inference_acc_num, calculated_list, predict_text_list, label_length_list, predict_length_list
