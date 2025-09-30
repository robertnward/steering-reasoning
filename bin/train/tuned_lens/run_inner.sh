# sleep infinity

num_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "NUM_GPUS: ${num_gpus}"

accelerate launch \
    --num_processes $num_gpus \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m steering_reasoning.train.tuned_lens \
        --config_path $CONFIG_PATH \
        --dataset_type $DATASET \
        --layer_idx $LAYER_IDX
