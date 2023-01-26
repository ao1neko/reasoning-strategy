#!/bin/bash

project_dir=$1
save_model_dir=$2



CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=no_reasoning_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_no_reasoning_step.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_no_reasoning_step.jsonl"  \
        --test_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_no_reasoning_step.jsonl"  \
        --model_name=bart \
        --load_model_dir="facebook/bart-large" \
        --train=true \
        --seed="42" \
        --predict=false \
        --train_epochs=100 \
        --eval_steps=1000  \
        --save_steps=1000 \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/bart_large/no_reasoning_step" \
        --run_dir="42/pretrain/bart_large/no_reasoning_step" \
        --batch_size=32 \
        --learning_rate=0.000001 \
        --load_tokenizer="facebook/bart-large"