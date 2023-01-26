#!/bin/bash

project_dir=$1
save_model_dir=$2



CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python "${project_dir}/src/main.py" \
        --device=cuda \
        --architecture_name=step_by_step \
        --train_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_step_by_step.jsonl" \
        --valid_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_step_by_step.jsonl" \
        --test_dataset_name="${project_dir}/numerical_datasets/data/pretrain/pretrain_test_step_by_step.jsonl" \
        --model_name=bart \
        --load_model_dir="${save_model_dir}/reasoning_model/42/pretrain/bart/step_by_step/best_model" \
        --train=false \
        --predict=true \
        --train_epochs=1 \
        --eval_steps=1  \
        --output_dir="${save_model_dir}/reasoning_model/42/pretrain/bart/step_by_step_test" \
        --run_dir="42/pretrain/bart/step_by_step_test" \
        --batch_size=64 \
        --load_tokenizer="facebook/bart-base"


