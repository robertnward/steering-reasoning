bash bin/helpers/modify_bias.sh

python3 -m steering_reasoning.helpers.pair_single --config_path $CONFIG_PATH --index1 $INDEX1 --index2 $INDEX2 --steering_vectors_path $STEERING_VECTORS_PATH

bash bin/eval/vanilla/run_inner.sh