bash bin/helpers/modify_bias.sh
cp -r /root/.cache/huggingface/datasets/ /workspace/hf_cache/

# sleep infinity
poetry run python3 -m steering_reasoning.metrics.match_effect --config_path $CONFIG_PATH