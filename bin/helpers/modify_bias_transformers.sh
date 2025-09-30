# CHANGE IN TRANSFORMERS
# qwen2
sed -i 's/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)/g' \
    /usr/local/lib/python3.11/dist-packages/transformers/models/qwen2/modeling_qwen2.py
# llama3
sed -i 's/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)/g' \
    /usr/local/lib/python3.11/dist-packages/transformers/models/llama/modeling_llama.py
# olmo2
sed -i 's/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)/g' \
    /usr/local/lib/python3.11/dist-packages/transformers/models/olmo2/modeling_olmo2.py
sed -i 's/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)/self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)/g' \
    /usr/local/lib/python3.11/dist-packages/transformers/models/olmo/modeling_olmo.py