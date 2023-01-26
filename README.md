> conda env create -n reasoning_step -f reasoning_step.yaml

# Make data
> cd numerical_datasets
> ./make_pretrain_data.sh
> ./make_data.sh
> ./concat_data.sh

# pretrain 
> cd scripts/simple
> ./all_at_once/pretrain_all_at_once.sh [project_dir_path] [save_dir_path]
> ./step_by_step/pretrain_step_by_step.sh [project_dir_path] [save_dir_path]
> ./token_by_token/pretrain_token_by_token.sh [project_dir_path] [save_dir_path]
example
> ./step_by_step/pretrain_step_by_step.sh /home/multi_reasoning_inference /work
model is saved at [save_dir_path]/reasoning_model/42/pretrain/step_by_step

# train
> ./all_at_once/train_all_at_once_shortest.sh [project_dir_path] [load_dir_path] [save_dir_path] 
> ./step_by_step/train_step_by_step_shortest.sh [project_dir_path] [load_dir_path] [save_dir_path] 
> ./token_by_token/train_token_by_token_shortest.sh [project_dir_path] [load_dir_path] [save_dir_path] 
example
> ./step_by_step/train_step_by_step_shortest.sh /home/multi_reasoning_inference /work/reasoning_model/42/pretrain/step_by_step/check-point-x /work 
(check-point-x : replace x with your best models)
model is saved at [save_dir_path]/reasoning_model/42/depth_1_5_distractor_3/step_by_step_shortest /work 

# load
> ./all_at_once/load_all_at_once_shortest.sh [project_dir_path] [load_dir_path] [save_dir_path]
> ./step_by_step/load_step_by_step_shortest.sh [project_dir_path] [load_dir_path] [save_dir_path]
> ./token_by_token/load_token_by_token_shortest.sh [project_dir_path] [load_dir_path] [save_dir_path]
example
> ./step_by_step/load_step_by_step_shortest.sh /home/multi_reasoning_inference /work/reasoning_model/42/depth_1_5_distractor_3/step_by_step_shortest/check-point-x /work 
output files are saved at [save_dir_path]/reasoning_model/42/depth_[test_depth]_distractor_3_by_depth_1_5_distractor_3/step_by_step_shortest
