# steered_hf_generic.py
import functools
import types
from typing import Any, Optional, Type

import torch
import torch.nn as nn


def _resolve_submodule(root: nn.Module, path: str) -> nn.Module:
    cur: Any = root
    for part in path.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    if not isinstance(cur, nn.Module):
        raise ValueError(f"Resolved object at '{path}' is not an nn.Module.")
    return cur


def _unpatch(target: nn.Module) -> None:
    """Restore a previously patched module's original forward, if any."""
    if getattr(target, "_steering_patched", False) and hasattr(
        target, "_steering_orig_forward"
    ):
        target.forward = target._steering_orig_forward  # type: ignore[attr-defined]
    if hasattr(target, "_steering_orig_forward"):
        delattr(target, "_steering_orig_forward")
    if hasattr(target, "_steering_patched"):
        delattr(target, "_steering_patched")


def _unpatch_path(module_owner: nn.Module, path: str) -> None:
    """Unpatch helper when we only know the path string."""
    try:
        target = _resolve_submodule(module_owner, path)
        _unpatch(target)
    except Exception:
        # If the path no longer exists or wasn't patched, ignore silently for minimal disruption.
        pass


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

    # Store original forward so we can unpatch later
    target._steering_orig_forward = orig_forward  # type: ignore[attr-defined]
    target.forward = types.MethodType(patched, target)
    target._steering_patched = True  # type: ignore[attr-defined]


# defers optional second patch block (commented) unchanged


def get_hf_wrapper(cls: Type[nn.Module]) -> Type[nn.Module]:
    """
    Returns a subclass of `cls` that:
      - adds `self.steering_vector` (shape [config.hidden_size])
      - reads `config.add_place` or accepts add_place kwarg via from_pretrained
      - injects the vector after that module
      - saves/loads cleanly via save_pretrained()/from_pretrained()

    Minimal additions:
      - `set_add_place(new_place: str)` to move the hook after loading.
      - `set_steering_vector(tensor)` to update the vector conveniently.
    """

    class Wrapper(cls):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            hidden_size = getattr(self.config, "hidden_size", None)
            if hidden_size is None:
                raise ValueError("config.hidden_size must be set.")
            self.steering_vector = nn.Parameter(torch.zeros(hidden_size))

            # Track current patch location so we can move it later
            self._steering_place: Optional[str] = getattr(
                self.config, "add_place", None
            )

            def _vec():
                p = next(self.parameters())
                return self.steering_vector.to(device=p.device, dtype=p.dtype)

            self._steering_vec_getter = _vec

            if self._steering_place:
                _patch_after(self, self._steering_place, self._steering_vec_getter)

        def set_add_place(self, new_place: str) -> None:
            """Move the steering hook to a new module path after loading."""
            if not isinstance(new_place, str) or not new_place:
                raise ValueError(
                    "new_place must be a non-empty string path to a submodule."
                )
            if self._steering_place:
                _unpatch_path(self, self._steering_place)
            _patch_after(self, new_place, self._steering_vec_getter)
            self._steering_place = new_place
            # Keep config in sync so save_pretrained/from_pretrained round-trips
            self.config.add_place = new_place

        def set_steering_vector(self, new_vec: torch.Tensor) -> None:
            """Update the steering vector (device/dtype/shape are handled)."""
            if not isinstance(new_vec, torch.Tensor):
                raise TypeError("new_vec must be a torch.Tensor")
            if new_vec.numel() != self.steering_vector.numel():
                raise ValueError(
                    f"steering_vector has wrong size: got {new_vec.numel()}, expected {self.steering_vector.numel()}"
                )
            with torch.no_grad():
                p = next(self.parameters())
                self.steering_vector.copy_(
                    new_vec.to(device=p.device, dtype=p.dtype).view_as(
                        self.steering_vector
                    )
                )

        def get_steering_vectors(self):
            return torch.tile(self.steering_vector.unsqueeze(0), (4, 1))

    return Wrapper
