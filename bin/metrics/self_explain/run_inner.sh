# sleep infinity

poetry run python3 -m steering_reasoning.metrics.self_explain \
    --config_path configs/metrics/self_explain.yml \
    --steering_vectors_path /from_s3/steering_vectors/$STEERING_VECTORS_PATH

# poetry run python3 -m steering_reasoning.metrics.self_explain --config_path configs/metrics/self_explain.yml --steering_vectors_path /from_s3/steering_vectors/$STEERING_VECTOR_PATH