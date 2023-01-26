#!/bin/bash

file_list="train_no_reasoning_step.jsonl valid_no_reasoning_step.jsonl test_no_reasoning_step.jsonl test_all_at_once_backward.jsonl train_all_at_once_backward.jsonl valid_all_at_once_backward.jsonl train_all_at_once_exhaustive.jsonl test_all_at_once_exhaustive.jsonl valid_all_at_once_exhaustive.jsonl test_all_at_once_shortest.jsonl train_all_at_once_shortest.jsonl valid_all_at_once_shortest.jsonl test_step_by_step_backward.jsonl train_step_by_step_backward.jsonl valid_step_by_step_backward.jsonl test_step_by_step_exhaustive.jsonl train_step_by_step_exhaustive.jsonl valid_step_by_step_exhaustive.jsonl test_step_by_step_shortest.jsonl train_step_by_step_shortest.jsonl valid_step_by_step_shortest.jsonl train_all_at_once_dot.jsonl valid_all_at_once_dot.jsonl test_all_at_once_dot.jsonl  train_step_by_step_dot.jsonl valid_step_by_step_dot.jsonl test_step_by_step_dot.jsonl  train_token_by_token_shortest.jsonl valid_token_by_token_shortest.jsonl test_token_by_token_shortest.jsonl train_token_by_token_exhaustive.jsonl valid_token_by_token_exhaustive.jsonl test_token_by_token_exhaustive.jsonl train_token_by_token_backward.jsonl valid_token_by_token_backward.jsonl test_token_by_token_backward.jsonl"
NL_file_list="train_NL_no_reasoning_step.jsonl valid_NL_no_reasoning_step.jsonl test_NL_no_reasoning_step.jsonl test_NL_all_at_once_backward.jsonl train_NL_all_at_once_backward.jsonl valid_NL_all_at_once_backward.jsonl train_NL_all_at_once_exhaustive.jsonl test_NL_all_at_once_exhaustive.jsonl valid_NL_all_at_once_exhaustive.jsonl test_NL_all_at_once_shortest.jsonl train_NL_all_at_once_shortest.jsonl valid_NL_all_at_once_shortest.jsonl test_NL_tep_by_step_backward.jsonl train_NL_step_by_step_backward.jsonl valid_NL_step_by_step_backward.jsonl test_NL_step_by_step_exhaustive.jsonl train_NL_step_by_step_exhaustive.jsonl valid_NL_step_by_step_exhaustive.jsonl test_NL_step_by_step_shortest.jsonl train_NL_step_by_step_shortest.jsonl valid_NL_step_by_step_shortest.jsonl train_NL_all_at_once_dot.jsonl valid_NL_all_at_once_dot.jsonl test_NL_all_at_once_dot.jsonl  train_NL_step_by_step_dot.jsonl valid_NL_step_by_step_dot.jsonl test_NL_step_by_step_dot.jsonl  train_NL_token_by_token_shortest.jsonl valid_NL_token_by_token_shortest.jsonl test_NL_token_by_token_shortest.jsonl train_NL_token_by_token_exhaustive.jsonl valid_NL_token_by_token_exhaustive.jsonl test_NL_token_by_token_exhaustive.jsonl train_NL_token_by_token_backward.jsonl valid_NL_token_by_token_backward.jsonl test_NL_token_by_token_backward.jsonl"
concat_file_list="depth_1_distractor_3 depth_2_distractor_3 depth_3_distractor_3 depth_4_distractor_3 depth_5_distractor_3"
output_file="./data/depth_1_5_distractor_3"
pretrain_dir="./data/pretrain"
mkdir -p $output_file

make_str_file_list() {
    string=""
    for concat_file in $concat_file_list; do
        old_string=$string
        string+="./data/${concat_file}/$1 "
    done
    echo $string
}

for file in $file_list; do
    str_file_list=$(make_str_file_list ${file})
    cat $str_file_list >$output_file/$file
    echo "create $output_file${file}"
done

for file in $NL_file_list; do
    str_file_list=$(make_str_file_list ${file})
    cat $str_file_list >$output_file/$file
    echo "create $output_file${file}"
done

cat $pretrain_dir/pretrain_no_reasoning_step.jsonl >>$output_file/train_no_reasoning_step.jsonl
cat $pretrain_dir/pretrain_all_at_once.jsonl >>$output_file/train_all_at_once_shortest.jsonl
cat $pretrain_dir/pretrain_all_at_once.jsonl >>$output_file/train_all_at_once_backward.jsonl
cat $pretrain_dir/pretrain_all_at_once.jsonl >>$output_file/train_all_at_once_exhaustive.jsonl
cat $pretrain_dir/pretrain_step_by_step.jsonl >>$output_file/train_step_by_step_shortest.jsonl
cat $pretrain_dir/pretrain_step_by_step.jsonl >>$output_file/train_step_by_step_backward.jsonl
cat $pretrain_dir/pretrain_step_by_step.jsonl >>$output_file/train_step_by_step_exhaustive.jsonl
cat $pretrain_dir/pretrain_token_by_token.jsonl >>$output_file/train_token_by_token_shortest.jsonl
cat $pretrain_dir/pretrain_token_by_token.jsonl >>$output_file/train_token_by_token_exhaustive.jsonl
cat $pretrain_dir/pretrain_token_by_token.jsonl >>$output_file/train_token_by_token_backward.jsonl

cat $pretrain_dir/pretrain_no_reasoning_step.jsonl >>$output_file/train_NL_no_reasoning_step.jsonl
cat $pretrain_dir/pretrain_all_at_once.jsonl >>$output_file/train_NL_all_at_once_shortest.jsonl
cat $pretrain_dir/pretrain_all_at_once.jsonl >>$output_file/train_NL_all_at_once_backward.jsonl
cat $pretrain_dir/pretrain_all_at_once.jsonl >>$output_file/train_NL_all_at_once_exhaustive.jsonl
cat $pretrain_dir/pretrain_step_by_step.jsonl >>$output_file/train_NL_step_by_step_shortest.jsonl
cat $pretrain_dir/pretrain_step_by_step.jsonl >>$output_file/train_NL_step_by_step_backward.jsonl
cat $pretrain_dir/pretrain_step_by_step.jsonl >>$output_file/train_NL_step_by_step_exhaustive.jsonl
cat $pretrain_dir/pretrain_token_by_token.jsonl >>$output_file/train_NL_token_by_token_shortest.jsonl
cat $pretrain_dir/pretrain_token_by_token.jsonl >>$output_file/train_NL_token_by_token_exhaustive.jsonl
cat $pretrain_dir/pretrain_token_by_token.jsonl >>$output_file/train_NL_token_by_token_backward.jsonl

echo "add pretrain data"
