#!/bin/bash

project_dir=$1
save_model_dir=$2



CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=all_at_once \
        --train_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_all_at_once.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_all_at_once.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_all_at_once.jsonl" \
        --model_name=t5 \
        --load_model_dir="${save_model_dir}/reasoning_model/42/pretrain/t5_large/all_at_once/best_model" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/t5_large/all_at_once_test" \
        --run_dir="42/pretrain/t5_large/all_at_once_test" \
        --batch_size=64 \
        --load_tokenizer="t5-large"


