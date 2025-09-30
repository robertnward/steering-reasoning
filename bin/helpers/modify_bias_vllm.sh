# allow to load partial weights (required when bias is added when it was False)
sed -i 's/if weights_not_loaded/if False and weights_not_loaded/g' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/model_loader/loader.py

# CHANGE IN VLLM
# qwen2
sed -E -i '
/^\s*self\.down_proj\s*=\s*RowParallelLinear\s*\(/,/^\s*\)\s*$/{
    s/\bbias=False\b/bias=True/
}
' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/models/qwen2.py
# llama3
sed -E -i '
/^\s*self\.down_proj\s*=\s*RowParallelLinear\s*\(/,/^\s*\)\s*$/{
    s/\bbias=bias\b/bias=True/
}
' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/models/llama.py
# olmo2
sed -E -i '
/^\s*self\.down_proj\s*=\s*RowParallelLinear\s*\(/,/^\s*\)\s*$/{
    s/\bbias=False\b/bias=True/
}
' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/models/olmo2.py
sed -E -i '
/^\s*self\.down_proj\s*=\s*RowParallelLinear\s*\(/,/^\s*\)\s*$/{
    s/\bbias=False\b/bias=True/
}
' /usr/local/lib/python3.11/dist-packages/vllm/model_executor/models/olmo.py
