export PYTHONPATH=/workspace:$PYTHONPATH
export VLLM_NO_USAGE_STATS=1

python3 -m steering_reasoning.helpers.modify_config \
    --model_dir /from_s3/model/ \
    --cfg_path config.json \
    --proj_type $PROJ_TYPE \
    --head_idx $HEAD_IDX \
    --path_patch_mode $PATH_PATCH_MODE

bash bin/helpers/modify_bias.sh

bash bin/eval/vanilla/run_inner.sh