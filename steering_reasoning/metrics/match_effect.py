import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
import vllm
from accelerate.utils import set_seed
from datasets import load_dataset
from loguru import logger
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from vllm import RequestOutput

from steering_reasoning.metrics.logit_lens import top_tokens_table
from steering_reasoning.train.rl.policy_model import tokenize_example
from steering_reasoning.train.rl.ray_workers.vllm_worker_wrap import (
    stateless_init_process_group,
)
from steering_reasoning.train.rl.trainer import is_answer_present
from steering_reasoning.utils.utils import set_logger
from steering_reasoning.visualize.pair_layers import _plot_grid_paper


@dataclass
class Config:
    seed: int

    model_path: str
    steering_vectors_path: str
    pair_accuracies_path: str
    dataset_path: str

    steering_perf: float

    num_samples: int

    num_samples_to_plot: int

    stop_token: str
    max_seq_length: int
    generation_temperature: float
    template_type: str

    output_dir: str
    verbose: bool

    def __post_init__(self):
        model_name = os.path.basename(self.model_path)
        self.output_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(self.output_dir, exist_ok=True)

        assert self.template_type in ["r1", "qwen_math"], self.template_type


def get_stop_token_id(tokenizer: PreTrainedTokenizer, stop_token: str):
    stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)
    assert len(stop_token_id) == 1, (
        f"Stop token must be a single token, instead {stop_token_id}"
    )
    stop_token_id = stop_token_id[0]

    return stop_token_id


def get_ds(config: Config, tokenizer):
    ds = load_dataset(config.dataset_path)["train"]
    ds = ds.shuffle(seed=config.seed)
    ds = ds.filter(lambda x: len(x["answer"]) > 0)
    if config.num_samples is not None:
        ds = ds.select(list(range(config.num_samples)))
    ds = ds.map(
        tokenize_example(
            tokenizer=tokenizer,
            stop_token=config.stop_token,
            max_seq_length=config.max_seq_length,
            template_type=config.template_type,
            append_to=False,
        ),
        batched=False,
        num_proc=None,
        remove_columns=["answer", "solution"],
    )

    return ds


def get_hidden_diff(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    steering_vector: torch.Tensor,
    num_layers: int,
    layer_idx: int,
):
    hiddens_dict: Dict[str, torch.Tensor] = {}

    def hiddens_capture(name):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output = output[0]
                hiddens_dict[name] = output.squeeze(0)

        return hook

    hiddens_handles = []
    for li in range(num_layers):
        layer = model.model.layers[li]
        hiddens_handle = layer.register_forward_hook(hiddens_capture(f"layer_{li}"))
        hiddens_handles.append(hiddens_handle)

    # Steered Hiddens
    zero_out_steering_vectors(model)
    model.model.layers[layer_idx].mlp.down_proj.bias.data = steering_vector.clone()
    model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
    )

    all_hiddens_steered = []
    for layer_idx_ in range(num_layers):
        all_hiddens_steered.append(
            hiddens_dict[f"layer_{layer_idx_}"][:-1].unsqueeze(0)
        )

    all_hiddens_steered = torch.concat(all_hiddens_steered, dim=0)

    # Vanilla Hiddens
    zero_out_steering_vectors(model)
    model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
    )

    all_hiddens_vanilla = []
    for layer_idx_ in range(num_layers):
        all_hiddens_vanilla.append(
            hiddens_dict[f"layer_{layer_idx_}"][:-1].unsqueeze(0)
        )

    all_hiddens_vanilla = torch.concat(all_hiddens_vanilla, dim=0)

    for hiddens_handle in hiddens_handles:
        hiddens_handle.remove()

    hidden_diff = all_hiddens_steered - all_hiddens_vanilla

    return hidden_diff


@torch.inference_mode()
def push_param(
    vllm_comm, vllm_actor, name: str, tensor: torch.Tensor, timeout_s: float = 120.0
):
    """
    Trainer-side: avoid deadlock by running the RPC in a background thread,
    then broadcasting from the main thread.
    """
    # 1) Kick off the blocking RPC on a separate thread.
    exc = {}

    def _rpc():
        try:
            vllm_actor.collective_rpc(
                "update_weight",
                args=(name, tensor.dtype, tensor.shape),
            )
        except Exception as e:
            exc["rpc_err"] = e

    t = threading.Thread(target=_rpc, daemon=True)
    t.start()

    # 2) Immediately do the broadcast on the *tensor's device/stream*.
    with torch.cuda.device(tensor.device):
        stream = torch.cuda.current_stream(device=tensor.device)
        vllm_comm.broadcast(tensor, src=0, stream=stream)
        # ensure the payload is fully sent before we proceed
        stream.synchronize()

    # 3) Wait for the RPC to finish (worker loads the weight after its broadcast returns).
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(
            "update_weight RPC did not return in time (likely worker stalled)."
        )
    if "rpc_err" in exc:
        raise RuntimeError(f"update_weight RPC failed: {exc['rpc_err']}")


def vllm_insert_steering_vector(
    vllm_comm, vllm_actor, steering_vector, layer_idx: int, num_layers: int
):
    for li in range(num_layers):
        push_param(
            vllm_comm=vllm_comm,
            vllm_actor=vllm_actor,
            name=f"model.layers.{li}.mlp.down_proj.bias",
            tensor=torch.zeros_like(steering_vector),
        )

    push_param(
        vllm_comm=vllm_comm,
        vllm_actor=vllm_actor,
        name=f"model.layers.{layer_idx}.mlp.down_proj.bias",
        tensor=steering_vector,
    )


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def init_process_group(vllm_actor, trainer_cuda_device=0, timeout_s=60):
    master_addr = "127.0.0.1"  # or this machine’s IP
    master_port = _free_port()
    world_size = 2  # trainer(0) + vLLM workers(1..tp)

    # 1) Kick off the worker init in a background thread (this call is blocking)
    exc = {}

    def _worker_init():
        try:
            vllm_actor.collective_rpc(
                "init_weight_update_group",
                args=(master_addr, master_port, 1, world_size),
            )
        except Exception as e:
            exc["worker"] = e

    t = threading.Thread(target=_worker_init, daemon=True)
    t.start()

    # (tiny pause lets workers spin up before rank 0 appears; not strictly required)
    time.sleep(0.05)

    trainer_comm = stateless_init_process_group(
        master_address=master_addr,
        master_port=master_port,
        world_size=world_size,
        rank=0,
        device=torch.device("cuda:1"),
    )

    # 3) Wait for the worker thread to finish (with a timeout for safety)
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError("Workers did not finish init_weight_update_group in time.")
    if "worker" in exc:
        raise RuntimeError(f"Worker init failed: {exc['worker']}")

    return trainer_comm


def zero_out_steering_vectors(model):
    for layer in model.model.layers:
        torch.nn.init.zeros_(layer.mlp.down_proj.bias)


def get_cossims(hidden_diff: torch.Tensor, steering_vectors: torch.Tensor):
    assert hidden_diff.shape[0] == steering_vectors.shape[0], (
        hidden_diff.shape,
        steering_vectors.shape,
    )

    num_layers = hidden_diff.shape[0]

    token_cossims = []
    diff_cossims = []

    for layer_steering_vector, layer_hidden_diff in zip(steering_vectors, hidden_diff):
        token_cossim = torch.nn.functional.cosine_similarity(
            layer_hidden_diff, layer_steering_vector.unsqueeze(0), dim=-1
        )

        diff_cossim = torch.nn.functional.cosine_similarity(
            layer_hidden_diff, layer_hidden_diff.mean(dim=0, keepdim=True), dim=-1
        )

        token_cossims.append(token_cossim)
        diff_cossims.append(diff_cossim)

    token_cossims = torch.vstack(token_cossims)
    diff_cossims = torch.vstack(diff_cossims)

    assert len(token_cossims) == num_layers, (len(token_cossims), num_layers)
    assert len(diff_cossims) == num_layers, (len(diff_cossims), num_layers)

    return token_cossims, diff_cossims


def save_cossims(mat, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    assert mat.shape[0] == mat.shape[1], mat.shape
    norm = Normalize(vmin=np.nanmin(mat), vmax=1)

    plt.figure()
    pm = plt.pcolormesh(mat, shading="auto", norm=norm)
    plt.colorbar(pm)
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight", format="pdf")
    plt.close()

    if np.nanmin(mat) >= 0:
        cmap_obj = LinearSegmentedColormap.from_list(
            "white_orange", [(0.0, "white"), (1.0, "#F28E2B")]
        )
        norm = Normalize(vmin=0.0, vmax=np.nanmax(mat))

    else:
        cmap_obj = LinearSegmentedColormap.from_list(
            "blue_white_orange",
            [(0.0, "#56B4E9"), (0.5, "white"), (1.0, "#F28E2B")],
        )
        norm = TwoSlopeNorm(vmin=np.nanmin(mat), vcenter=0.0, vmax=1)

    _plot_grid_paper(
        mat,
        title="loh",
        filename=os.path.basename(savepath.replace(".pdf", "_lt")),
        norm=norm,
        savedir=Path(os.path.dirname(savepath)),
        stats_map=None,
        annotate_std=False,
        cmap=cmap_obj,
        dpi=200,
        N=mat.shape[0],
        marker_threshold=10,
    )

    with open(savepath.replace(".pdf", ".npy"), "wb") as f:
        np.save(f, mat)


def scatter_with_fit(
    y: np.ndarray, x: np.ndarray, xlabel: str, ylabel: str, savepath: str
):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    y_array = y[np.triu_indices_from(y, k=1)]
    x_array = x[np.triu_indices_from(x, k=1)]

    assert y_array.shape == x_array.shape, (y_array.shape, x_array.shape)
    assert y_array.ndim == 1 and x_array.ndim == 1, (y_array.shape, x_array.shape)

    r = float(np.corrcoef(x_array, y_array)[0, 1])

    plt.scatter(x_array, y_array, label=f"r = {r:.3f}")

    a, b = np.polyfit(x_array, y_array, 1)
    xline = np.linspace(x_array.min(), x_array.max(), 100)
    plt.plot(xline, a * xline + b, lw=1.2, label="fit", color="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(savepath, bbox_inches="tight")
    plt.close()


def hist_plots(cossims: np.ndarray, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    fig, axes = plt.subplots(4, 7, figsize=(18, 9), sharex=False, sharey=False)
    # fig.suptitle("Lora scalings (lora_A) values", fontsize=16, y=0.94)

    for layer_idx, ax in enumerate(axes.ravel()[: len(cossims)]):
        ax.hist(cossims[layer_idx], bins=50)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("CosSim", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath, dpi=150)
    plt.close()


def save_logit_lens(vectors, unembed_matrix, tokenizer, savedir: str):
    all_cossims = []
    for layer_idx, steering_vector in enumerate(vectors):
        dot_sims = unembed_matrix @ steering_vector

        cos_sims = (
            unembed_matrix / torch.linalg.norm(unembed_matrix, dim=-1, keepdim=True)
        ) @ (steering_vector / torch.linalg.norm(steering_vector))
        all_cossims.append(cos_sims.float().cpu().numpy())
        _, top_tokens_cos_sims_idx = torch.topk(cos_sims, 50)
        top_tokens_cos_sims = [tokenizer.decode(t) for t in top_tokens_cos_sims_idx]

        top_tokens_table(
            top_tokens_idx=top_tokens_cos_sims_idx.cpu().numpy(),
            top_tokens=top_tokens_cos_sims,
            cos_sims=cos_sims.float().cpu().numpy(),
            dot_sims=dot_sims.float().cpu().numpy(),
            savepath=os.path.join(savedir, "top", f"layer_{layer_idx}.csv"),
        )

        _, top_tokens_cos_sims_idx = torch.topk(-cos_sims, 50)
        top_tokens_cos_sims = [tokenizer.decode(t) for t in top_tokens_cos_sims_idx]

        top_tokens_table(
            top_tokens_idx=top_tokens_cos_sims_idx.cpu().numpy(),
            top_tokens=top_tokens_cos_sims,
            cos_sims=cos_sims.float().cpu().numpy(),
            dot_sims=dot_sims.float().cpu().numpy(),
            savepath=os.path.join(savedir, "bottom", f"layer_{layer_idx}.csv"),
        )

    all_cossims = np.array(all_cossims)
    hist_plots(cossims=all_cossims, savepath=os.path.join(savedir, "cossims.pdf"))


@pyrallis.wrap()
def run(config: Config):
    set_seed(seed=config.seed, device_specific=False, deterministic=False)
    set_logger(config.verbose)
    torch.set_grad_enabled(False)

    vllm_actor = vllm.LLM(
        model=config.model_path,
        trust_remote_code=True,
        seed=config.seed,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=config.max_seq_length,
        max_seq_len_to_capture=config.max_seq_length * 2,
        dtype="bfloat16",
        enable_lora=False,
        tensor_parallel_size=1,
        worker_extension_cls="steering_reasoning.train.rl.ray_workers.vllm_worker_wrap.WorkerWrap",
    )
    vllm_comm = init_process_group(vllm_actor=vllm_actor)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = model.eval().to("cuda:1")
    zero_out_steering_vectors(model)

    num_layers = model.config.num_hidden_layers

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    stop_token_id = get_stop_token_id(tokenizer=tokenizer, stop_token=config.stop_token)

    sampling_params = vllm.SamplingParams(
        n=1,
        repetition_penalty=1.0,
        top_p=1.0,
        top_k=-1,
        temperature=config.generation_temperature,
        stop_token_ids=[stop_token_id],
        seed=config.seed,
    )
    ds = get_ds(config=config, tokenizer=tokenizer)

    # Put handles on lora-scaling weights
    steering_vectors = np.load(config.steering_vectors_path)
    steering_vectors = (
        torch.from_numpy(steering_vectors).to(model.device).to(model.dtype)
    )
    steering_vectors = steering_vectors[:num_layers]
    layer_lengths = defaultdict(list)
    layer_rewards = defaultdict(list)

    pair_accuracies = np.load(config.pair_accuracies_path)[:num_layers, :num_layers]

    diag = np.diag(pair_accuracies)
    pair_max = np.maximum(diag[:, None], diag[None, :])
    pair_accuracies_norm = (pair_accuracies - pair_max) / (
        config.steering_perf - pair_max
    )

    layer_sum_hidden_diffs: Dict[int, torch.Tensor] = defaultdict(
        lambda: torch.zeros(
            num_layers,
            model.config.hidden_size,
            dtype=torch.float32,
            device=model.device,
        )
    )
    layer_token_cossims: Dict[int, List[List[torch.Tensor]]] = defaultdict(
        lambda: torch.empty(num_layers, 0)
    )
    layer_diff_cossims: Dict[int, List[List[torch.Tensor]]] = defaultdict(
        lambda: torch.empty(num_layers, 0)
    )
    layer_num_tokens: Dict[int, int] = defaultdict(lambda: 0)

    for sample_idx in trange(0, len(ds), desc="Sample"):
        all_token_cossims = []
        all_diff_cossims = []

        prompt_token_ids = ds["prompt_input_ids"][sample_idx]
        prompt_attention_mask = ds["prompt_attention_mask"][sample_idx]
        answer_token_ids = ds["answer_input_ids"][sample_idx]

        sampling_params.max_tokens = config.max_seq_length - len(prompt_attention_mask)

        for layer_idx in range(num_layers):
            vllm_insert_steering_vector(
                vllm_comm=vllm_comm,
                vllm_actor=vllm_actor,
                steering_vector=steering_vectors[layer_idx],
                layer_idx=layer_idx,
                num_layers=num_layers,
            )
            request_outputs: List[RequestOutput] = vllm_actor.generate(
                prompt_token_ids=[prompt_token_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            assert len(request_outputs) == 1 and len(request_outputs[0].outputs) == 1, (
                len(request_outputs),
                len(request_outputs[0].outputs),
            )
            generation = request_outputs[0].outputs[0]
            generation_token_ids = generation.token_ids[: sampling_params.max_tokens]

            is_correct, _ = is_answer_present(
                processing_class=tokenizer,
                answer_token_ids=answer_token_ids,
                generation_token_ids=generation_token_ids,
                template_type=config.template_type,
            )
            reward = bool(is_correct)

            prompt_decoded = tokenizer.decode(prompt_token_ids)
            logger.debug(f"Prompt: {prompt_decoded}")
            generation_decoded = tokenizer.decode(generation_token_ids)
            logger.debug(f"Generation: {generation_decoded}")

            input_ids = torch.tensor(
                prompt_token_ids + generation_token_ids,
                device=model.device,
                dtype=torch.long,
            )[:-1]
            attention_mask = torch.ones_like(input_ids).to(model.device)
            position_ids = torch.arange(len(input_ids)).to(model.device)

            hidden_diff = get_hidden_diff(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                steering_vector=steering_vectors[layer_idx],
                num_layers=num_layers,
                layer_idx=layer_idx,
            )

            layer_sum_hidden_diffs[layer_idx] += hidden_diff.to(torch.float32).sum(
                dim=1
            )

            assert input_ids.dim() == 1, input_ids.shape
            layer_num_tokens[layer_idx] += len(input_ids)

            token_cossims, diff_cossims = get_cossims(
                hidden_diff=hidden_diff, steering_vectors=steering_vectors
            )
            layer_token_cossims[layer_idx] = torch.hstack(
                [layer_token_cossims[layer_idx], token_cossims.float().cpu()]
            )
            layer_diff_cossims[layer_idx] = torch.hstack(
                [layer_diff_cossims[layer_idx], diff_cossims.float().cpu()]
            )

            all_token_cossims.append(token_cossims.mean(1).float().cpu().numpy())
            all_diff_cossims.append(diff_cossims.mean(1).float().cpu().numpy())

            layer_lengths[layer_idx].append(len(input_ids))
            layer_rewards[layer_idx].append(reward)

            logger.info(
                f"Sample {sample_idx}; Layer {layer_idx}; Len: {input_ids.shape}"
            )

        layer_lengths_mean = {
            layer_idx: np.mean(lengths) for layer_idx, lengths in layer_lengths.items()
        }
        layer_rewards_mean = {
            layer_idx: np.mean(rewards) for layer_idx, rewards in layer_rewards.items()
        }

        logger.info(
            f"Layer Lengths Mean after {sample_idx} samples: {layer_lengths_mean} and rewards: {layer_rewards_mean}"
        )

        all_token_cossims = np.vstack(all_token_cossims)
        all_diff_cossims = np.vstack(all_diff_cossims)

        indices = np.arange(num_layers).astype(float)
        layer_dists = np.abs(indices[None, :] - indices[:, None])
        layer_dists /= layer_dists.max()

        if sample_idx > config.num_samples_to_plot:
            continue

        for name, cossims in zip(
            ["token_cossims", "diff_cossims", "dists"],
            [all_token_cossims, all_diff_cossims, layer_dists],
        ):
            save_cossims(
                mat=cossims,
                savepath=os.path.join(
                    config.output_dir, name, f"sample_{sample_idx}.pdf"
                ),
            )
            scatter_with_fit(
                y=pair_accuracies,
                x=cossims,
                xlabel=name,
                ylabel="Pairwise Accuracy",
                savepath=os.path.join(
                    config.output_dir,
                    f"{name}_vs_pair_accuracies",
                    f"sample_{sample_idx}.png",
                ),
            )
            scatter_with_fit(
                y=pair_accuracies_norm,
                x=cossims,
                xlabel=name,
                ylabel="Pairwise Normalized Accuracy",
                savepath=os.path.join(
                    config.output_dir,
                    f"{name}_vs_pair_accuracies_norm",
                    f"sample_{sample_idx}.png",
                ),
            )

            if name != "dists":
                scatter_with_fit(
                    y=cossims,
                    x=layer_dists,
                    xlabel="Layer Dists",
                    ylabel="Cosine Similarity",
                    savepath=os.path.join(
                        config.output_dir,
                        f"{name}_vs_dists",
                        f"sample_{sample_idx}.png",
                    ),
                )

    mean_layer_cossims = []
    mean_token_cossims = []
    mean_diff_cossims = []
    for layer_idx in range(num_layers):
        mean_diff = layer_sum_hidden_diffs[layer_idx] / layer_num_tokens[layer_idx]
        mean_diff_cossim = torch.nn.functional.cosine_similarity(
            mean_diff, steering_vectors
        )

        mean_layer_cossims.append(mean_diff_cossim.unsqueeze(0).cpu())
        mean_token_cossims.append(layer_token_cossims[layer_idx].mean(1))
        mean_diff_cossims.append(layer_diff_cossims[layer_idx].mean(1))

    mean_layer_cossims = torch.vstack(mean_layer_cossims)
    mean_token_cossims = torch.vstack(mean_token_cossims)
    mean_diff_cossims = torch.vstack(mean_diff_cossims)

    save_cossims(
        mat=mean_layer_cossims,
        savepath=os.path.join(config.output_dir, "layer_cossims", "all_samples.pdf"),
    )
    save_cossims(
        mat=mean_token_cossims,
        savepath=os.path.join(config.output_dir, "token_cossims", "all_samples.pdf"),
    )
    save_cossims(
        mat=mean_diff_cossims,
        savepath=os.path.join(config.output_dir, "diff_cossims", "all_samples.pdf"),
    )

    last_diffs = torch.vstack(
        [
            (layer_sum_hidden_diffs[layer_idx] / layer_num_tokens[layer_idx])[-1]
            for layer_idx in range(num_layers)
        ]
    )

    unembed_matrix = model.get_output_embeddings().weight.data.cuda()
    save_logit_lens(
        vectors=last_diffs.to("cuda:1").to(torch.float32),
        unembed_matrix=unembed_matrix.to("cuda:1").to(torch.float32),
        tokenizer=tokenizer,
        savedir=os.path.join(config.output_dir, "logit_lens"),
    )

    last_diffs_norm = last_diffs / last_diffs.norm(dim=1, keepdim=True)
    last_diff_cossim_matrix = last_diffs_norm @ last_diffs_norm.T
    last_diff_cossim_matrix = last_diff_cossim_matrix.float().cpu().numpy()
    np.save(
        os.path.join(config.output_dir, "last_diff_cossim_matrix.npy"),
        last_diff_cossim_matrix,
    )


if __name__ == "__main__":
    run()
