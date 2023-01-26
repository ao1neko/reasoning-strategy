#!/bin/bash
for inference_step in $(seq 1 12); do
        distractor_num=3
        equation_num=$(($inference_step + $distractor_num))

        PYTHONHASHSEED=0 python "./make_dataset.py" \
                --train_data_size=1000 \
                --valid_data_size=200 \
                --test_data_size=200 \
                --inference_step=$inference_step \
                --equation_num=$equation_num \
                --output_dir="./data"

        echo "create depth ${inference_step} data"
        declare -a method=("train" "valid" "test")
        for i in {0..2}; do
                PYTHONHASHSEED=0 python "./convert_numerical_data.py" \
                        --input_file="./data/depth_${inference_step}_distractor_${distractor_num}/${method[$i]}.jsonl" \
                        --method=${method[$i]}
                echo "convert depth ${inference_step} data to depth ${inference_step} ${method[$i]}_data"
        done
done
