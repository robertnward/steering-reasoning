pip install circuitsvis==1.43.3

if [[ $PATCHED_MODEL_AUTO_MAP == *Llama* ]]; then
    cp steering_reasoning/models/head_patched_path/llama/transformers.py $MODEL_PATH/patched_model.py
elif [[ $PATCHED_MODEL_AUTO_MAP == *Qwen* ]]; then
    cp steering_reasoning/models/head_patched_path/qwen/transformers.py $MODEL_PATH/patched_model.py
fi

python3 -m steering_reasoning.helpers.modify_config \
    --model_dir $MODEL_PATH \
    --cfg_path config.json \
    --proj_type $PROJ_TYPE \
    --head_idx $HEAD_IDX \
    --auto_map_causal_lm $PATCHED_MODEL_AUTO_MAP \
    --path_patch_layer_index $PATH_PATCH_LAYER_INDEX \
    --patch_path $PATCH_PATH \
    --add_layer $ADD_LAYER

bash bin/helpers/modify_bias.sh

sed -i 's/attn_output = self.o_proj(attn_output)/attn_output = self.o_proj(attn_output) if make_o_proj else attn_output/' /usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py

sed -Ei '/^[[:space:]]*past_key_value[s]?:[^\n]*, *$/{n;s/^([[:space:]]*)(cache_position:[^\n]*, *$)/\1\2\
\1make_o_proj: bool = True,/;}' /usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py

# sleep infinity
python3 -m steering_reasoning.metrics.pre_last_layer_steering \
    --config_path $CONFIG_PATH \
    --model_path $MODEL_PATH

# python3 -m steering_reasoning.metrics.pre_last_layer_steering --config_path configs/metrics/pre_last_layer_steering/llama3.1-8b-instruct.yml --model_path $MODEL_PATH