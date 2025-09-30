# steered_hf_generic.py
import functools
import types
from typing import Any, Type

import torch
import torch.nn as nn


def _resolve_submodule(root: nn.Module, path: str) -> nn.Module:
    cur: Any = root
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

    @functools.wraps(orig_forward)
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

        @functools.wraps(extra_orig_forward)
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


def get_hf_wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Returns a subclass of `cls` that:
      - adds `self.steering_vector` (shape [config.hidden_size])
      - reads `config.add_place` or accepts add_place kwarg via from_pretrained
      - injects the vector after that module
      - saves/loads cleanly via save_pretrained()/from_pretrained()
    """

    class Wrapper(cls):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("config.hidden_size must be set.")
            self.steering_vector = nn.Parameter(torch.zeros(hidden_size))

            def _vec():
                p = next(self.parameters())
                return self.steering_vector.to(device=p.device, dtype=p.dtype)

            _patch_after(self, self.config.add_place, _vec)

        def get_steering_vectors(self):
            return torch.tile(self.steering_vector.unsqueeze(0), (4, 1))

    return Wrapper
