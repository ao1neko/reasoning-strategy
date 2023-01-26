#!/bin/bash

project_dir=$1
save_model_dir=$2



CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=token_by_token \
        --train_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_token_by_token.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_token_by_token.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_test_token_by_token.jsonl" \
        --model_name=bart \
        --load_model_dir="${save_model_dir}/reasoning_model/42/pretrain/bart_large/token_by_token/best_model" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/bart_large/token_by_token_test" \
        --run_dir="42/pretrain/bart_large/token_by_token_test" \
        --batch_size=64 \
        --load_tokenizer="facebook/bart-large"


