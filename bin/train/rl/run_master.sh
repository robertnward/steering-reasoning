bash bin/helpers/train_bash_commands.sh

if [ -n "${ADD_PLACE:-}" ]; then
    export PYTHONPATH=/workspace:$PYTHONPATH
    export VLLM_NO_USAGE_STATS=1
    sed -i 's/if weights_not_loaded/if False and weights_not_loaded/g' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/model_loader/loader.py
fi

ray start --head --port=6379

# sleep infinity
poetry run python -m steering_reasoning.train.rl.main --config_path $CONFIG_PATH --seed $SEED --is_dist $IS_DIST
