export INIT_PYTHONPATH=$PYTHONPATH
export PYTHONPATH=/workspace:$PYTHONPATH
export VLLM_NO_USAGE_STATS=1

python3 -m steering_reasoning.helpers.modify_config \
    --model_dir /from_s3/model/ \
    --cfg_path config.json \
    --new_arch $MODEL_ARCH

bash bin/eval/vanilla/run_inner.sh