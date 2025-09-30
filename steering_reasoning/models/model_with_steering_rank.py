import torch
import torch.nn as nn


def get_wrapper(cls: nn.Module, steering_rank: int):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.steering_matrix_A = nn.Parameter(
                torch.empty(
                    len(self.model.layers),
                    steering_rank,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self.steering_matrix_B = nn.Parameter(
                torch.empty(
                    steering_rank,
                    self.config.hidden_size,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            self._patch_layers()

        def get_steering_vectors(self):
            return self.steering_matrix_A @ self.steering_matrix_B

        def _patch_layers(self):
            """Wrap MLP forward to inject dynamic bias."""

            def make_patched_forward(orig_forward, layer_idx):
                def patched_forward(*args, **kwargs):
                    out = orig_forward(*args, **kwargs)
                    # Add dynamic bias here
                    steering_vectors = self.get_steering_vectors()
                    out = out + steering_vectors[layer_idx]

                    return out

                return patched_forward

            for idx, layer in enumerate(self.model.layers):
                layer.mlp.down_proj.forward = make_patched_forward(
                    layer.mlp.down_proj.forward, idx
                )

    return Wrapper
