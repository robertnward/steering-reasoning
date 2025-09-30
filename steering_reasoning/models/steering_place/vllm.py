# steered_vllm_arch_names.py
# Register steered variants under ORIGINAL architecture names:
#   "LlamaForCausalLM" and "QwenForCausalLM"
# Import this module BEFORE constructing vLLM's LLM(...).

import types

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.llama import LlamaForCausalLM as VLLM_LlamaForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM as VLLM_Qwen2ForCausalLM


def _resolve_submodule(root: nn.Module, path: str) -> nn.Module:
    cur = root
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise ValueError(f"Resolved object at '{path}' is not an nn.Module.")
    return cur


def _patch_after(module_owner: nn.Module, add_place: str, get_vector):
    target = _resolve_submodule(module_owner, add_place)
    if getattr(target, "_steering_patched", False):
        return
    orig_forward = target.forward

    def patched(_self_module, *args, **kwargs):
        out = orig_forward(*args, **kwargs)
        vec = get_vector()

        def _add_bias(t: torch.Tensor) -> torch.Tensor:
            shape = [1] * (t.dim() - 1) + [-1]
            return t + vec.view(*shape)

        if isinstance(out, torch.Tensor):
            return _add_bias(out)
        if isinstance(out, (list, tuple)) and out and isinstance(out[0], torch.Tensor):
            x = _add_bias(out[0])
            return type(out)((x, *out[1:]))
        raise RuntimeError(
            f"Patched module at '{add_place}' returned unsupported type: {type(out)}"
        )

    target.forward = types.MethodType(patched, target)
    target._steering_patched = True

    if "input_layernorm" in add_place:
        extra_add_place = add_place.replace("input_layernorm", "self_attn")
    elif "post_attention_layernorm" in add_place:
        extra_add_place = add_place.replace("post_attention_layernorm", "mlp")
    else:
        extra_add_place = None

    if extra_add_place is not None:
        extra_target = _resolve_submodule(module_owner, extra_add_place)
        if getattr(extra_target, "_steering_patched", False):
            return
        extra_orig_forward = extra_target.forward

        def patched(_self_module, *args, **kwargs):
            out = extra_orig_forward(*args, **kwargs)
            vec = get_vector()

            def _add_bias(t: torch.Tensor) -> torch.Tensor:
                shape = [1] * (t.dim() - 1) + [-1]
                return t + vec.view(*shape)

            if isinstance(out, torch.Tensor):
                return _add_bias(out)
            if (
                isinstance(out, (list, tuple))
                and out
                and isinstance(out[0], torch.Tensor)
            ):
                x = _add_bias(out[0])
                return type(out)((x, *out[1:]))
            raise RuntimeError(
                f"Patched module at '{add_place}' returned unsupported type: {type(out)}"
            )

        extra_target.forward = types.MethodType(patched, extra_target)
        extra_target._steering_patched = True


class SteeredLlamaForCausalLM(VLLM_LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if not getattr(self.model.config, "add_place", None):
            raise ValueError("config.add_place must be set in the saved config.")
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("config.hidden_size must be set.")

        # Parameter name MUST match HF wrapper for state-dict parity
        self.steering_vector = nn.Parameter(torch.zeros(hidden_size))

        def _vec():
            p = next(self.parameters())
            return self.steering_vector.to(device=p.device, dtype=p.dtype)

        _patch_after(self, self.model.config.add_place, _vec)


class SteeredQwen2ForCausalLM(VLLM_Qwen2ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        if not getattr(self.model.config, "add_place", None):
            raise ValueError("config.add_place must be set in the saved config.")
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("config.hidden_size must be set.")

        self.steering_vector = nn.Parameter(torch.zeros(hidden_size))

        def _vec():
            p = next(self.parameters())
            return self.steering_vector.to(device=p.device, dtype=p.dtype)

        _patch_after(self, self.model.config.add_place, _vec)
