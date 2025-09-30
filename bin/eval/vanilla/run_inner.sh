
if $IS_STEERING; then
    echo "It is steering. Modifying the files"
    bash bin/helpers/modify_bias.sh
fi

if [[ -v INIT_PYTHONPATH ]]; then
    export SET_PYTHONPATH=$PYTHONPATH
    export PYTHONPATH=$INIT_PYTHONPATH
fi
num_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [[ -v INIT_PYTHONPATH ]]; then
    export PYTHONPATH=$SET_PYTHONPATH
fi

mkdir -p logs

for split_id in $(seq 0 $((num_gpus - 1))); do
    echo "$split_id/$num_gpus"
    
    if [[ -e /from_s3/adapter_model ]]; then
        echo "It is a lora"
        CUDA_VISIBLE_DEVICES=$split_id \
            python3 -m steering_reasoning.eval.eval \
            --config_path $CONFIG_PATH \
            --seed $SEED \
            --adapter_path /from_s3/adapter_model \
            --split_id $split_id \
            --total_splits $num_gpus >> logs/logs_${split_id}.txt 2>&1 &
    else
        echo "It is not a lora"
        if [ -n "${TEMP:-}" ]; then
            CUDA_VISIBLE_DEVICES=$split_id \
                python3 -m steering_reasoning.eval.eval \
                --config_path $CONFIG_PATH \
                --seed $SEED \
                --top_p $TOPP \
                --generation_temperature $TEMP \
                --split_id $split_id \
                --total_splits $num_gpus >> logs/logs_${split_id}.txt 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$split_id \
                python3 -m steering_reasoning.eval.eval \
                --config_path $CONFIG_PATH \
                --seed $SEED \
                --split_id $split_id \
                --total_splits $num_gpus >> logs/logs_${split_id}.txt 2>&1 &
        fi
    fi
done

wait 


poetry run python3 -m steering_reasoning.eval.merge_pkls --savedir results_splits/
