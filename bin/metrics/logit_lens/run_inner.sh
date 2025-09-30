
# sleep infinity
poetry run python3 -m steering_reasoning.metrics.logit_lens \
    --config_path configs/metrics/logit_lens.yml \
    --model_path $MODEL_PATH \
    --steering_vectors_dir $STEERING_VECTORS_DIR