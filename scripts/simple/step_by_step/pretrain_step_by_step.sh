#!/bin/bash

project_dir=$1
save_model_dir=$2


CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=step_by_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_step_by_step.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_step_by_step.jsonl"  \
        --test_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_test_step_by_step.jsonl"  \
        --model_name=t5 \
        --load_model_dir="t5-base" \
        --train=true \
        --seed="42" \
        --predict=false \
        --train_epochs=30 \
        --eval_steps=100  \
        --save_steps=100 \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/step_by_step" \
        --run_dir="42/pretrain/step_by_step" \
        --batch_size=64 \
        --learning_rate=0.001 \
        --load_tokenizer="t5-base"