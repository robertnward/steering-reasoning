from typing import Optional

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM


def _unregister_attention(layer):
    """Remove the old layer's Attention blocks from vLLM's global registry."""
    ctx = get_current_vllm_config().compilation_config.static_forward_context
    for m in layer.modules():
        if hasattr(m, "layer_name"):  # every Attention has this attr
            ctx.pop(m.layer_name, None)


class LlamaLastLayer(LlamaDecoderLayer):
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__(
            config, cache_config=cache_config, quant_config=quant_config, prefix=prefix
        )
        # Your extra learnable parameter
        self.proj_type = config.proj_type
        self.head_idx = config.head_idx

        self.steering_vector = torch.nn.Parameter(torch.zeros(config.hidden_size))

    def get_after_input_layernorm(self, hidden_states, residual):
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        return hidden_states, residual

    def get_qkv(self, hidden_states):
        # ATTN
        qkv, _ = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size],
            dim=-1,
        )

        return q, k, v

    def insert_patch(self, insert_what, insert_to):
        insert_to[
            :,
            self.head_idx * self.self_attn.head_dim : (self.head_idx + 1)
            * self.self_attn.head_dim,
        ] = insert_what[
            :,
            self.head_idx * self.self_attn.head_dim : (self.head_idx + 1)
            * self.self_attn.head_dim,
        ]
        return insert_to

    def __call__(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_vanilla, residual_vanilla = self.get_after_input_layernorm(
            hidden_states=hidden_states, residual=residual
        )
        q_vanilla, k_vanilla, v_vanilla = self.get_qkv(
            hidden_states=hidden_states_vanilla
        )

        hidden_states_steered, residual_steered = self.get_after_input_layernorm(
            hidden_states=hidden_states + self.steering_vector, residual=residual
        )
        q_steered, k_steered, v_steered = self.get_qkv(
            hidden_states=hidden_states_steered
        )
        if self.proj_type == "q_proj":
            assert q_steered.dim() == 2, q_steered.dim()

            q = self.insert_patch(insert_what=q_vanilla, insert_to=q_steered)
            k = k_steered
            v = v_steered
        elif self.proj_type == "inverse_q_proj":
            assert q_vanilla.dim() == 2, q_vanilla.dim()

            q = self.insert_patch(insert_what=q_steered, insert_to=q_vanilla)
            k = k_vanilla
            v = v_vanilla
        elif self.proj_type == "k_proj":
            assert k_steered.dim() == 2, k_steered.dim()

            q = q_steered
            k = self.insert_patch(insert_what=k_vanilla, insert_to=k_steered)
            v = v_steered
        elif self.proj_type == "inverse_k_proj":
            assert k_vanilla.dim() == 2, k_vanilla.dim()

            q = q_vanilla
            k = self.insert_patch(insert_what=k_steered, insert_to=k_vanilla)
            v = v_vanilla
        elif self.proj_type == "v_proj":
            assert v_steered.dim() == 2, v_steered.dim()

            q = q_steered
            k = k_steered
            v = self.insert_patch(insert_what=v_vanilla, insert_to=v_steered)
        elif self.proj_type == "inverse_v_proj":
            assert v_vanilla.dim() == 2, v_vanilla.dim()

            q = q_vanilla
            k = k_vanilla
            v = self.insert_patch(insert_what=v_steered, insert_to=v_vanilla)
        elif self.proj_type == "all_v_proj":
            assert v_steered.dim() == 2, v_steered.dim()

            q = q_steered
            k = k_steered
            v = v_vanilla
        elif self.proj_type == "inverse_all_v_proj":
            assert v_vanilla.dim() == 2, v_vanilla.dim()

            q = q_vanilla
            k = k_vanilla
            v = v_steered
        elif self.proj_type in ["skip_attn", "skip_layer", "base"]:
            q = q_vanilla
            k = k_vanilla
            v = v_vanilla
        elif self.proj_type == "no_patch":
            q = q_steered
            k = k_steered
            v = v_steered
        else:
            raise ValueError(f"Unknown proj_type {self.proj_type}")

        q, k = self.self_attn.rotary_emb(positions, q, k)
        attn_output = self.self_attn.attn(q, k, v)
        hidden_states, _ = self.self_attn.o_proj(attn_output)

        # Fully Connected
        if self.proj_type in ["skip_layer", "base"]:
            residual = residual_vanilla
        else:
            residual = residual_steered

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if self.proj_type == "skip_layer":
            hidden_states = hidden_states + self.steering_vector

        return hidden_states, residual


class LlamaForCausalLM_CustomLast(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        layers = self.model.layers
        last = len(layers) - 1
        old = layers[last]

        _unregister_attention(old)

        cfg = self.model.config
        cache_cfg = vllm_config.cache_config
        quant_cfg = vllm_config.quant_config

        layers[last] = LlamaLastLayer(
            cfg,
            cache_config=cache_cfg,
            quant_config=quant_cfg,
            prefix=f"model.layers.{last}",
        )

    def load_weights(self, weights):
        loaded_weights = super().load_weights(weights)

        self.model.layers[-1].steering_vector = torch.nn.Parameter(
            self.model.layers[-2].mlp.down_proj.bias.clone()
        )
        self.model.layers[-2].mlp.down_proj.bias = torch.nn.Parameter(
            torch.zeros_like(self.model.layers[-2].mlp.down_proj.bias)
        )

        return loaded_weights
