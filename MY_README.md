> conda env create -n reasoning_step -f reasoning_step.yaml
or
> pyenv shell miniconda3-4.7.12
> activate reasoning_step

# Make data
README.md参照

# pretrain 
README.md参照
example
> ./pretrain_step_by_step.sh /home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_inference_ver4 /work02/aoki0903/

# train
README.md参照
example
> ./train_step_by_step_shortest.sh /home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_inference_ver4 /work02/aoki0903/reasoning_model/42/pretrain/step_by_step/checkpoint-6200 /work02/aoki0903/

# load
README.md参照
example
> ./load_step_by_step_shortest.sh /home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_inference_ver4 /work02/aoki0903/reasoning_model/42/depth_1_5_distractor_3/step_by_step_shortest/check-point-x /work02/aoki0903/


TODO
./train_all_at_once_shortest.sh /home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_inference_ver4 /work02/aoki0903/reasoning_model/42/pretrain/all_at_once/best_model /work02/aoki0903/

./train_step_by_step_shortest.sh /home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_inference_ver4 /work02/aoki0903/reasoning_model/42/pretrain/step_by_step/best_model /work02/aoki0903/

./train_token_by_token_shortest.sh /home/aoki0903/sftp_sync/MyPonderNet/multi_reasoning_inference_ver4 /work02/aoki0903/reasoning_model/42/pretrain/token_by_token/best_model /work02/aoki0903/
