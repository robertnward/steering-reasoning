import difflib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open  # streaming, no full load
from transformers import AutoModelForCausalLM


def load_param_from_safetensors_dir(repo_dir: str, parameter_name: str):
    """
    Return the torch.Tensor for `parameter_name` from a directory saved by save_pretrained().
    Searches for *.safetensors.index.json first (sharded), then *.safetensors (single-file),
    recursively across subdirectories.
    """
    # 1) Look for index files (sharded checkpoints)
    index_filenames = {
        "model.safetensors.index.json",
        "pytorch_model.safetensors.index.json",
        "adapter_model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "optimizer.safetensors.index.json",
    }

    for root, _, files in os.walk(repo_dir):
        for f in files:
            if f in index_filenames:
                index_path = os.path.join(root, f)
                with open(index_path, "r", encoding="utf-8") as fh:
                    idx = json.load(fh)
                weight_map = idx.get("weight_map", {})
                if parameter_name in weight_map:
                    shard_rel = weight_map[parameter_name]
                    shard_path = (
                        os.path.join(root, shard_rel)
                        if not os.path.isabs(shard_rel)
                        else shard_rel
                    )
                    with safe_open(shard_path, framework="pt", device="cpu") as sf:
                        return sf.get_tensor(parameter_name)
                # keep searching other index files

    # 2) No index hit: scan all .safetensors files and check their keys
    candidate_files = []
    for root, _, files in os.walk(repo_dir):
        for f in files:
            if f.endswith(".safetensors"):
                candidate_files.append(os.path.join(root, f))

    if not candidate_files:
        raise FileNotFoundError(f"No .safetensors files found under: {repo_dir}")

    for path in candidate_files:
        with safe_open(path, framework="pt", device="cpu") as sf:
            if parameter_name in sf.keys():
                return sf.get_tensor(parameter_name)

    # 3) Not found: offer close matches to help debug
    sample_keys = []
    try:
        with safe_open(candidate_files[0], framework="pt", device="cpu") as sf:
            sample_keys = list(sf.keys())
    except Exception:
        pass

    close = difflib.get_close_matches(parameter_name, sample_keys, n=8)
    raise KeyError(f"Parameter {parameter_name!r} not found. Close matches: {close}")


# --- usage ---
# tensor = load_param_from_safetensors_dir("/path/to/save_pretrained/dir", "parameter_name")
# print(tensor.shape, tensor.dtype)
# tensor = tensor.to("cuda")  # if you want it on GPU


@torch.no_grad()
def run2():
    savedir = "/workspace/result/"
    os.makedirs(savedir, exist_ok=True)
    model_26 = AutoModelForCausalLM.from_pretrained(
        "/from_s3/model_26/", trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    # steering_vector_26 = model_26.model.layers[26].mlp.down_proj.bias.data
    # steering_vector_27_self_attn = load_param_from_safetensors_dir(
    #     "/from_s3/model_27_self_attn/", "steering_vector"
    # )
    # head_dim = model_26.model.layers[27].self_attn.head_dim
    # heads_per_group = (
    #     model_26.config.num_attention_heads // model_26.config.num_key_value_heads
    # )
    # ln_steering_vector_26 = model_26.model.layers[27].input_layernorm(
    #     steering_vector_26
    # )

    # # model_26.model.layers[27].input_layernorm(steering_vector_26)
    # hidden_shape = (1, 1, -1, head_dim)
    # value_states = (
    #     (model_26.model.layers[27].self_attn.v_proj.weight.data @ steering_vector_26)
    #     .view(hidden_shape)
    #     .transpose(1, 2)
    # )[0, 1, 0]
    # value_states = torch.tile(value_states, (heads_per_group, 1)).reshape(-1)

    # proj = (
    #     model_26.model.layers[27]
    #     .self_attn.o_proj.weight[
    #         heads_per_group * head_dim : 2 * heads_per_group * head_dim
    #     ]
    #     .T
    #     @ value_states
    # )
    # ov_26 = proj @ steering_vector_26

    # steering_vector_after_ov_26 = steering_vector_26 + ov_26

    # cossim = torch.nn.functional.cosine_similarity(
    #     steering_vector_27_self_attn, steering_vector_after_ov_26, dim=0
    # )

    # print("cossim", cossim)

    model = model_26

    lns = [
        model.model.layers[i].input_layernorm.weight.data
        for i in range(len(model.model.layers))
    ]

    for idx, ln in enumerate(lns):
        ln = ln.float().cpu().numpy()

        plt.hist(ln)
        plt.savefig(os.path.join(savedir, f"ln_{idx}.png"))

    nrows, ncols = 4, 7
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3, nrows * 2.5), sharex=True
    )
    axes = axes.ravel()  # 28 axes

    for idx, ln in enumerate(lns):  # len(lns) == 28
        arr = ln.float().cpu().numpy()
        axes[idx].hist(arr)
        axes[idx].set_title(f"ln_{idx}", fontsize=9)
        axes[idx].set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "ln_grid.png"), dpi=200)
    plt.close(fig)

    nrows, ncols = 4, 7
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 3, nrows * 2.5), sharex=True
    )
    axes = axes.ravel()  # 28 axes

    for idx, ln in enumerate(lns):  # len(lns) == 28
        arr = ln.float().cpu().numpy()
        arr = arr[arr < 1]
        axes[idx].hist(arr)
        axes[idx].set_title(f"ln_{idx}", fontsize=9)
        axes[idx].set_yscale("log")

    plt.tight_layout()
    fig.savefig(os.path.join(savedir, "ln_grid_less_1.png"), dpi=200)
    plt.close(fig)


@torch.no_grad()
def run():
    savedir = "/workspace/result/"
    os.makedirs(savedir, exist_ok=True)

    steering_vector = load_param_from_safetensors_dir(
        "/from_s3/model/", "steering_vector"
    )
    steering_vector = steering_vector.float().cpu().numpy()

    np.save(os.path.join(savedir, "steering_vector.npy"), steering_vector)


if __name__ == "__main__":
    run()
