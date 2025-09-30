import math

import numpy as np
import torch
import torch.nn as nn
import torchtt as tt


def closest_factors(n: int) -> tuple[int, int]:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    a = int(math.isqrt(n))  # floor(sqrt(n))
    while a > 0 and n % a != 0:  # walk downward until a divides n
        a -= 1

    return (a, n // a)


def get_wrapper(cls: nn.Module, tt_rank: int, tt_d: int = 2):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.out_core_shapes = closest_factors(n=self.model.layers)
            self.in_core_shapes = closest_factors(n=self.config.hidden_size)

            ranks = [1] + [tt_rank] * (tt_d - 1) + [1]
            self.cores = []
            for k in range(tt_d):
                shape = (
                    ranks[k],
                    self.out_core_shapes[k],
                    self.in_core_shapes[k],
                    ranks[k + 1],
                )
                # glorot-like variance scaling
                scale = 1.0 / math.sqrt(
                    self.out_core_shapes[k] * self.in_core_shapes[k]
                )
                core = nn.Parameter(
                    scale * torch.randn(shape, dtype=self.dtype, device=self.device)
                )
                self.cores.append(core)

            with torch.no_grad():
                self.init_steering_vectors = self.get_steering_vectors()
            self.init_steering_vectors.requires_grad = False

            self._patch_layers()

        def get_steering_vectors(self):
            steering_vectors = tt.TT(self.cores).full()
            steering_vectors = steering_vectors.reshape(
                np.prod(self.out_core_shapes), np.prod(self.in_core_shapes)
            )
            return steering_vectors

        def _patch_layers(self):
            """Wrap MLP forward to inject dynamic bias."""

            def make_patched_forward(orig_forward, layer_idx):
                def patched_forward(*args, **kwargs):
                    out = orig_forward(*args, **kwargs)
                    # Add dynamic bias here
                    steering_vectors = self.get_steering_vectors()

                    # +- so that steering_vectors on the first step are none
                    out = (
                        out
                        + steering_vectors[layer_idx]
                        - self.init_steering_vectors[layer_idx]
                    )

                    return out

                return patched_forward

            for idx, layer in enumerate(self.model.layers):
                layer.mlp.down_proj.forward = make_patched_forward(
                    layer.mlp.down_proj.forward, idx
                )

    return Wrapper
