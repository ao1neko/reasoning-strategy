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
        --load_model_dir="facebook/bart-base" \
        --train=true \
        --seed="42" \
        --predict=false \
        --train_epochs=30 \
        --eval_steps=100  \
        --save_steps=100 \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/bart/no_reasoning_step" \
        --run_dir="42/pretrain/bart/no_reasoning_step" \
        --batch_size=64 \
        --learning_rate=0.0001 \
        --load_tokenizer="facebook/bart-base"