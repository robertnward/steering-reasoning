

num_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())")

mkdir logs

for split_id in $(seq 0 $((num_gpus - 1))); do
    CUDA_VISIBLE_DEVICES=$split_id \
        poetry run python3 -m steering_reasoning.data_gen.gen_completions_ds \
        --config_path $CONFIG_PATH \
        --split_id $split_id \
        --total_splits $num_gpus >> logs/logs_${split_id}.txt 2>&1 &
done

wait

poetry run python3 -m steering_reasoning.data_gen.merge_datasets --savedir results/