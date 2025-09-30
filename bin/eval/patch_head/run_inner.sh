export VLLM_NO_USAGE_STATS=1

if [[ $PATCHED_MODEL_AUTO_MAP == *Llama* ]]; then
    cp steering_reasoning/models/head_patched_path/llama/transformers.py /from_s3/model/patched_model.py
elif [[ $PATCHED_MODEL_AUTO_MAP == *Qwen* ]]; then
    cp steering_reasoning/models/head_patched_path/qwen/transformers.py /from_s3/model/patched_model.py
fi

python3 -m steering_reasoning.helpers.modify_config \
    --model_dir /from_s3/model/ \
    --cfg_path config.json \
    --new_arch $ARCH \
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


bash bin/eval/vanilla/run_inner.sh