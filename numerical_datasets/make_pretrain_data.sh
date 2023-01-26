#!/bin/bash

PYTHONHASHSEED=0 python "./make_pretrain_data.py" \
        --seed=10

echo "create pretrain data"

cp "./data/pretrain/pretrain.jsonl" "./data/pretrain/pretrain_test.jsonl"
PYTHONHASHSEED=0 python "./convert_numerical_data.py" \
        --input_file="./data/pretrain/pretrain.jsonl" \
        --method="train"
PYTHONHASHSEED=0 python "./convert_numerical_data.py" \
        --input_file="./data/pretrain/pretrain_test.jsonl" \
        --method="test"
echo "convert pretrain_data"

mv "./data/pretrain/pretrain_all_at_once_shortest.jsonl" "./data/pretrain/pretrain_all_at_once.jsonl"
mv "./data/pretrain/pretrain_step_by_step_shortest.jsonl" "./data/pretrain/pretrain_step_by_step.jsonl"
mv "./data/pretrain/pretrain_test_step_by_step_shortest.jsonl" "./data/pretrain/pretrain_test_step_by_step.jsonl"
mv "./data/pretrain/pretrain_token_by_token_shortest.jsonl" "./data/pretrain/pretrain_token_by_token.jsonl"
mv "./data/pretrain/pretrain_test_token_by_token_shortest.jsonl" "./data/pretrain/pretrain_test_token_by_token.jsonl"
rm "./data/pretrain/pretrain_all_at_once_backward.jsonl" "./data/pretrain/pretrain_all_at_once_exhaustive.jsonl" "./data/pretrain/pretrain_step_by_step_backward.jsonl" "./data/pretrain/pretrain_step_by_step_exhaustive.jsonl" "./data/pretrain/pretrain_test_all_at_once_shortest.jsonl" "./data/pretrain/pretrain_test_all_at_once_exhaustive.jsonl" "./data/pretrain/pretrain_test_all_at_once_backward.jsonl" "./data/pretrain/pretrain_test_no_reasoning_step.jsonl" "./data/pretrain/pretrain_test_step_by_step_exhaustive.jsonl" "./data/pretrain/pretrain_test_step_by_step_backward.jsonl" "./data/pretrain/pretrain_test_token_by_token_exhaustive.jsonl" "./data/pretrain/pretrain_test_token_by_token_backward.jsonl" "./data/pretrain/pretrain_token_by_token_exhaustive.jsonl" "./data/pretrain/pretrain_token_by_token_backward.jsonl"

echo "rmove old data"
