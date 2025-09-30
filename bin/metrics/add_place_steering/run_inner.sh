export PYTHONPATH=/workspace:$PYTHONPATH
export VLLM_NO_USAGE_STATS=1
pip install circuitsvis

python3 -m steering_reasoning.helpers.modify_config \
    --model_dir $MODEL_PATH \
    --cfg_path config.json \
    --new_arch $MODEL_ARCH


# sleep infinity
python3 -m steering_reasoning.metrics.add_place_steering \
    --config_path $CONFIG_PATH \
    --model_path $MODEL_PATH

# python3 -m steering_reasoning.metrics.add_place_steering --config_path configs/metrics/add_place_steering/llama3.1-8b-instruct.yml --model_path $MODEL_PATH