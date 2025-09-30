
pip install circuitsvis==1.43.3
cp -r /root/.cache/huggingface/datasets/ /workspace/hf_cache/

# sleep infinity
poetry run python3 -m steering_reasoning.metrics.lora1_plots --config_path $CONFIG_PATH