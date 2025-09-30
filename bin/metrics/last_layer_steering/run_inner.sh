bash bin/helpers/modify_bias.sh

# sleep infinity
python3 -m steering_reasoning.metrics.last_layer_steering \
    --config_path $CONFIG_PATH \
    --model_path $MODEL_PATH

# python3 -m steering_reasoning.metrics.last_layer_steering --config_path configs/metrics/last_layer_steering.yml --model_path $MODEL_PATH