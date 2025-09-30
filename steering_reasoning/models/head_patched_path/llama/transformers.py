import copy
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as HF_LlamaDecoderLayer,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
)
from transformers.processing_utils import Unpack


class PatchedLlamaDecoderLayer(HF_LlamaDecoderLayer):
    """
    Drop-in replacement that keeps the HF decoder-layer signature and flow.

    Implements two attention passes (vanilla & steered). Exactly one pass uses
    the KV cache (reads & updates). The other computes attention against a
    **deep-copied** cache (so it can read history but won't mutate the real
    cache). Head-level patching is performed **pre o_proj**.

    Cache side is chosen by `self._kv_cache_side in {"vanilla","steered"}`.
    By default it's "vanilla"; you can optionally set `config.path_patch_cache_side`.

    No GQA: raises if num_key_value_heads != num_heads.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Controls from config
        self.head_idx = config.head_idx
        self.proj_type = config.proj_type
        self.patch_path = config.patch_path

        if self.proj_type in ["head", "inverse_head"]:
            assert self.patch_path in ["all", "residual", "mlp"], (
                f"patch_path={self.patch_path} for proj_type={self.proj_type}"
            )
        else:
            assert self.patch_path == "all", (
                f"patch_path={self.patch_path} for proj_type={self.proj_type}"
            )

        self.self_attn2 = copy.deepcopy(self.self_attn)
        self.self_attn2.layer_idx = config.num_hidden_layers - 1

        self.steering_vector = torch.nn.Parameter(torch.zeros(config.hidden_size))

    def insert_patch(self, insert_what, insert_to):
        h_, d_ = self.head_idx, self.self_attn.head_dim
        result = insert_to.clone()
        result[:, :, h_ * d_ : (h_ + 1) * d_] = insert_what[
            :, :, h_ * d_ : (h_ + 1) * d_
        ].clone()

        return result

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor]:
        residual_vanilla = hidden_states
        residual_steered = hidden_states + self.steering_vector
        hidden_states_vanilla = self.input_layernorm(hidden_states)
        hidden_states_steered = self.input_layernorm(
            hidden_states + self.steering_vector
        )

        # Self Attention
        # attention will be without o_proj -- see `sed` in bin/eval/patch_head_eval.sh
        hidden_states_vanilla, _ = self.self_attn(
            hidden_states=hidden_states_vanilla,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            make_o_proj=False,
            **kwargs,
        )
        hidden_states_steered, _ = self.self_attn2(
            hidden_states=hidden_states_steered,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            make_o_proj=False,
            **kwargs,
        )

        if self.proj_type == "head":
            hidden_states = self.insert_patch(
                insert_what=hidden_states_vanilla, insert_to=hidden_states_steered
            )
            residual = residual_steered
        elif self.proj_type == "inverse_head":
            hidden_states = self.insert_patch(
                insert_what=hidden_states_steered, insert_to=hidden_states_vanilla
            )
            residual = residual_steered
        elif self.proj_type in ["base", "skip_layer"]:
            hidden_states = hidden_states_vanilla
            residual = residual_vanilla
        elif self.proj_type == "skip_attn":
            hidden_states = hidden_states_vanilla
            residual = residual_steered
        elif self.proj_type == "no_patch":
            hidden_states = hidden_states_steered
            residual = residual_steered
        else:
            raise ValueError(f"Unknown proj_type: {self.proj_type}")

        hidden_states_vanilla = self.self_attn.o_proj(hidden_states_vanilla)
        hidden_states_steered = self.self_attn.o_proj(hidden_states_steered)
        hidden_states = self.self_attn.o_proj(hidden_states)

        if self.patch_path == "all":
            mlp_input = residual + hidden_states
            residual = residual + hidden_states
        elif self.patch_path == "residual":
            mlp_input = residual + hidden_states_steered
            residual = residual + hidden_states
        elif self.patch_path == "mlp":
            mlp_input = residual + hidden_states
            residual = residual + hidden_states_steered
        else:
            raise ValueError(f"Unknown patch_path: {self.patch_path}")

        # Fully Connected
        mlp_input = self.post_attention_layernorm(mlp_input)
        hidden_states = self.mlp(mlp_input)
        hidden_states = residual + hidden_states

        if self.proj_type == "skip_layer":
            hidden_states = hidden_states + self.steering_vector

        outputs = (hidden_states,)
        assert not output_attentions

        return outputs


class PatchedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        path_patch_layer_index = int(getattr(config, "path_patch_layer_index"))
        if not (0 <= path_patch_layer_index < len(self.model.layers)):
            raise IndexError(
                f"path_patch_layer_index {path_patch_layer_index} out of range 0..{len(self.model.layers) - 1}"
            )
        dev = next(self.model.layers[0].parameters()).device
        dt = next(self.model.layers[0].parameters()).dtype
        self.model.layers[path_patch_layer_index] = PatchedLlamaDecoderLayer(
            config, layer_idx=path_patch_layer_index
        ).to(dev, dt)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.model.layers = model.model.layers[:-1]

        layers = model.model.layers
        pre_last = layers[-2]
        if pre_last.mlp.down_proj.bias is None:
            raise RuntimeError("Expected pre-last mlp.down_proj to have bias.")
        vec = pre_last.mlp.down_proj.bias.detach().clone()
        pre_last.mlp.down_proj.bias.data.zero_()

        i = int(getattr(model.config, "path_patch_layer_index", 0))
        patched_layer: PatchedLlamaDecoderLayer = layers[i]  # type: ignore
        patched_layer.steering_vector.data.copy_(vec.to(patched_layer.steering_vector))

        patched_layer.self_attn2.load_state_dict(
            patched_layer.self_attn.state_dict(), strict=True
        )

        return model
