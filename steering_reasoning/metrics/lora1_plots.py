import copy
import json
import os
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
import vllm
from accelerate.utils import set_seed
from datasets import load_dataset
from loguru import logger
from peft import PeftModelForCausalLM
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from vllm import RequestOutput

from steering_reasoning.metrics.plots import (
    calc_stable_rank,
    draw_pairwise_cossims,
    draw_singular_values,
    get_logit_lens,
    save_logit_lens,
)
from steering_reasoning.metrics.pre_last_layer_steering import token_scores
from steering_reasoning.train.rl.policy_model import (
    guidance_prompt,
    tokenize_example,
)
from steering_reasoning.train.rl.trainer import is_answer_present
from steering_reasoning.utils.utils import pad_sequences, set_logger


@dataclass
class Config:
    seed: int

    model_path: str
    adapters_path: str

    dataset_path: str
    num_samples: Optional[int]

    max_seq_length: int
    generation_temperature: float
    stop_token: str
    template_type: str
    max_samples_to_plot: int

    output_dir: str
    verbose: bool

    def __post_init__(self):
        model_name = os.path.basename(self.model_path)
        self.output_dir = os.path.join(self.output_dir, model_name, f"seed-{self.seed}")
        os.makedirs(self.output_dir, exist_ok=True)

        assert self.template_type in ["r1", "qwen_math"], self.template_type


def save_single_lora_scaling(lora_scaling: List[np.ndarray], savedir: str):
    os.makedirs(savedir, exist_ok=True)

    for layer_idx in range(len(lora_scaling)):
        np.save(os.path.join(savedir, f"layer-{layer_idx}.npy"), lora_scaling)


def scalings_imshow_plot(lora_scalings: np.ndarray, borders: List[int], savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    # logger.info(f"Borders: {borders}")

    lora_scalings_cp = lora_scalings.copy()
    # flip = -(lora_scalings_cp.mean(1) > 0).astype(float)
    # lora_scalings_cp *= np.sign(lora_scalings.mean(-1, keepdims=True))
    lora_scalings_cp /= np.nanmax(lora_scalings_cp, axis=-1, keepdims=True) - np.nanmin(
        lora_scalings_cp, axis=-1, keepdims=True
    )
    plt.figure(figsize=(30, 5))

    norm = mpl.colors.SymLogNorm(
        linthresh=1.0, vmin=lora_scalings_cp.min(), vmax=lora_scalings_cp.max()
    )
    im = plt.imshow(lora_scalings_cp, norm=norm, cmap="berlin")
    plt.colorbar(
        im, label="lora_A value", location="top", orientation="horizontal", pad=0.05
    )
    for border in borders:
        plt.axvline(border - 0.5, color="green", linewidth=2)
    # plt.title("lora_A values")
    plt.xlabel("Sequence idx")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(savepath, dpi=500)
    plt.close()


def scalings_hist_plots(lora_scalings: List[np.ndarray], savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    fig, axes = plt.subplots(4, 7, figsize=(18, 9), sharex=False, sharey=False)
    fig.suptitle("Lora scalings (lora_A) values", fontsize=16, y=0.94)

    for layer_idx, ax in enumerate(axes.ravel()[: len(lora_scalings)]):
        ax.hist(lora_scalings[layer_idx], bins=50)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("lora_A value", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_yscale("log")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(savepath, dpi=150)
    plt.close()


def savetext(text, savepath):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    with open(savepath, "+w") as f:
        f.write(text)


def plot_scatters(X, Y, savepath):
    """
    X, Y: numpy arrays with shape (N, L)
    savepath: output PDF path, e.g. 'out.pdf'
    """
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.shape == Y.shape and X.ndim == 2, (
        f"X and Y must be 2D arrays of same shape [N, L], instead: {X.shape} vs {Y.shape}"
    )
    N, L = X.shape

    rows = 4
    cols = max(1, ceil(N / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    ax_list = axes.ravel()

    for i in range(N):
        ax = ax_list[i]
        xi, yi = X[i].copy(), Y[i].copy()
        if yi.mean() < 0:
            yi *= -1

        # Pearson r
        r = float(np.corrcoef(xi, yi)[0, 1]) if L > 1 else np.nan
        ax.scatter(
            xi,
            yi,
            s=10,
            alpha=0.85,
            label=f"r = {r:.3f}" if np.isfinite(r) else "r = nan",
        )

        # Best-fit line (least squares)
        if L >= 2 and np.std(xi) > 0:
            a, b = np.polyfit(xi, yi, 1)
            xline = np.linspace(xi.min(), xi.max(), 100)
            ax.plot(xline, a * xline + b, lw=1.2, label="fit", color="red")

        ax.set_title(f"Layer {i}")
        ax.legend(frameon=False)

    # Remove any unused axes
    for j in range(N, rows * cols):
        fig.delaxes(ax_list[j])

    fig.supxlabel("cossim", fontsize=11)  # Matplotlib ≥ 3.4
    fig.supylabel("magnitude", fontsize=11)

    fig.tight_layout()
    fig.savefig(savepath, format="pdf")
    plt.close(fig)


def get_peft_model(model, adapters: Dict[str, str]):
    name0, path0 = next(iter(adapters.items()))
    peft_model = PeftModelForCausalLM.from_pretrained(
        model, path0, adapter_name=name0, is_trainable=False
    )

    for name, path in list(adapters.items())[1:]:
        peft_model.load_adapter(path, adapter_name=name)

    peft_model.eval()

    return peft_model


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


def get_stop_token_id(tokenizer: PreTrainedTokenizer, stop_token: str):
    stop_token_id = tokenizer.encode(stop_token, add_special_tokens=False)
    assert len(stop_token_id) == 1, (
        f"Stop token must be a single token, instead {stop_token_id}"
    )
    stop_token_id = stop_token_id[0]

    return stop_token_id


def get_system_boundaries(tokenizer):
    system_len = len(
        tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": guidance_prompt}],
            add_generation_prompt=False,
            tokenize=True,
            padding=False,
            truncation=False,
            return_dict=True,
        )["input_ids"]
    )
    system_w_role_len = len(
        tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": guidance_prompt},
                {"role": "user", "content": ""},
            ],
            add_generation_prompt=False,
            tokenize=True,
            padding=False,
            truncation=False,
            return_dict=True,
        )["input_ids"]
    )

    return system_len, system_w_role_len


def get_steering_vectors(peft_model, num_layers) -> Tuple[List[float], List[float]]:
    steering_vectors = []
    steering_vector_norms = []
    for layer_idx in range(num_layers):
        peft_model.set_adapter(f"layer-{layer_idx}")

        layer = peft_model.model.model.layers[layer_idx]
        lora_B = (
            layer.mlp.down_proj.lora_B[f"layer-{layer_idx}"]
            .weight.data.squeeze(-1)
            .cpu()
            .numpy()
        )

        steering_vector_norm = np.linalg.norm(lora_B)
        steering_vector = lora_B / steering_vector_norm

        steering_vector_norms.append(steering_vector_norm)
        steering_vectors.append(steering_vector)

    return steering_vectors, steering_vector_norms


def get_scalings_and_hiddens(
    peft_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    layer_idx: int,
):
    lora_scaling_dict: Dict[str, torch.Tensor] = {}
    hiddens_dict: Dict[str, torch.Tensor] = {}

    def lora_scaling_capture(name):
        def hook(module, inputs, output):
            lora_scaling_dict[name] = output.squeeze(0).squeeze(1)

        return hook

    def hiddens_capture(name):
        def hook(module, inputs, output):
            hiddens_dict[name] = output.squeeze(0)

        return hook

    peft_model.set_adapter(f"layer-{layer_idx}")

    layer = peft_model.model.model.layers[layer_idx]
    lora_A = layer.mlp.down_proj.lora_A[f"layer-{layer_idx}"]
    scalings_handle = lora_A.register_forward_hook(
        lora_scaling_capture(f"layer_{layer_idx}")
    )
    hiddens_handle = layer.mlp.down_proj.base_layer.register_forward_hook(
        hiddens_capture(f"layer_{layer_idx}")
    )

    peft_model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
    )

    lora_scalings = lora_scaling_dict[f"layer_{layer_idx}"]
    hiddens = hiddens_dict[f"layer_{layer_idx}"]

    scalings_handle.remove()
    hiddens_handle.remove()

    return lora_scalings, hiddens


def plot_line(data, xlabel: str, ylabel: str, savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    with mpl.rc_context(
        {
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    ):
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(savepath, bbox_inches="tight", format="pdf")
        plt.close()


def save_token_acts(data: List[Dict[str, Any]], savepath: str) -> None:
    # Ensure parent directory exists
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)

    with open(savepath, "w", encoding="utf-8") as f:
        for row in data:
            # Write one JSON object per line
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


@pyrallis.wrap()
def run(config: Config):
    set_seed(seed=config.seed, device_specific=False, deterministic=False)
    set_logger(config.verbose)
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = model.eval().cuda()

    num_layers = model.config.num_hidden_layers

    adapters = {
        f"layer-{layer_idx}": os.path.join(
            config.adapters_path, f"lora-1-layer-{layer_idx}"
        )
        for layer_idx in range(num_layers)
    }

    peft_model = get_peft_model(model=model, adapters=adapters)

    vllm_actor = vllm.LLM(
        model=config.model_path,
        trust_remote_code=True,
        seed=config.seed,
        enable_prefix_caching=False,
        enforce_eager=False,
        max_model_len=config.max_seq_length,
        max_seq_len_to_capture=config.max_seq_length * 2,
        dtype="bfloat16",
        enable_lora=True,
    )
    lora_requests = [
        vllm.lora.request.LoRARequest(
            f"lora-1-layer{layer_idx}",
            layer_idx + 1,
            os.path.join(config.adapters_path, f"lora-1-layer-{layer_idx}"),
        )
        for layer_idx in range(num_layers)
    ]
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    stop_token_id = get_stop_token_id(tokenizer=tokenizer, stop_token=config.stop_token)

    system_len, system_w_role_len = get_system_boundaries(tokenizer)

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
    steering_vectors, steering_vector_norms = get_steering_vectors(
        peft_model=peft_model, num_layers=num_layers
    )

    draw_pairwise_cossims(
        steering_vectors=steering_vectors,
        savepath=os.path.join(
            config.output_dir, "steering_vectors", "pairwise_cossims.png"
        ),
    )
    draw_singular_values(
        steering_vectors=steering_vectors,
        savepath=os.path.join(
            config.output_dir, "steering_vectors", "singular_values.png"
        ),
    )
    stable_rank = calc_stable_rank(steering_vectors)
    logger.info(f"Stable rank: {stable_rank}")
    top_tokens_with_scores = get_logit_lens(
        ranking_type="prob",
        model=model,
        steering_vectors=steering_vectors,
        tokenizer=tokenizer,
    )
    save_logit_lens(
        top_tokens_with_scores=top_tokens_with_scores,
        savepath=os.path.join(config.output_dir, "steering_vectors", "logit_lens.txt"),
    )

    top_tokens_with_scores = get_logit_lens(
        ranking_type="cossim",
        model=model,
        steering_vectors=steering_vectors,
        tokenizer=tokenizer,
    )
    save_logit_lens(
        top_tokens_with_scores=top_tokens_with_scores,
        savepath=os.path.join(
            config.output_dir, "steering_vectors", "logit_lens_cossim.txt"
        ),
    )

    all_tokens = [[] for _ in range(num_layers)]
    all_lora_scalings = [torch.zeros(0) for _ in range(num_layers)]
    all_hidden_norms = [torch.zeros(0) for _ in range(num_layers)]
    for_json = []

    for sample_idx in trange(0, len(ds), desc="Sample"):
        logger.info(f"Sample: {sample_idx}")
        prompt_token_ids = ds["prompt_input_ids"][sample_idx]
        answer_token_ids = ds["answer_input_ids"][sample_idx]

        sampling_params.max_tokens = config.max_seq_length - len(prompt_token_ids)

        sample_lora_scalings = []
        sample_hidden_norms = []
        sample_cossims = []
        sample_tokens = []
        for layer_idx in range(num_layers):
            request_outputs: List[RequestOutput] = vllm_actor.generate(
                prompt_token_ids=[prompt_token_ids],
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=lora_requests[layer_idx],
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

            input_ids = torch.tensor(
                prompt_token_ids + generation_token_ids[:-1],  # remove eos
                device=peft_model.device,
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids).to(peft_model.device)
            position_ids = torch.arange(len(input_ids)).to(peft_model.device)

            # extract lora_A outputs after the pass

            lora_scalings, hiddens = get_scalings_and_hiddens(
                peft_model=peft_model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_idx=layer_idx,
            )

            hidden_norm_mean = torch.norm(hiddens, dim=-1).mean()

            steering_vector = (
                torch.from_numpy(steering_vectors[layer_idx])
                .to(hiddens.device)
                .to(hiddens.dtype)
                .unsqueeze(0)
            )

            cossims = torch.nn.functional.cosine_similarity(
                hiddens, steering_vector, dim=-1
            ).squeeze(0)
            sample_lora_scalings.append(
                lora_scalings * steering_vector_norms[layer_idx]
            )
            sample_hidden_norms.append(hidden_norm_mean)
            sample_cossims.append(cossims)
            sample_tokens.append(input_ids)

            all_lora_scalings[layer_idx] = torch.hstack(
                [
                    all_lora_scalings[layer_idx],
                    (lora_scalings * steering_vector_norms[layer_idx]).float().cpu(),
                ]
            )
            all_hidden_norms[layer_idx] = torch.hstack(
                [all_hidden_norms[layer_idx], torch.norm(hiddens, dim=-1).float().cpu()]
            )
            all_tokens[layer_idx] += [
                tokenizer.decode(token_id, skip_special_tokens=False)
                for token_id in input_ids
            ]

            for_json.append(
                {
                    "sample_idx": sample_idx,
                    "layer_idx": layer_idx,
                    "tokens": [
                        tokenizer.decode(token_id, skip_special_tokens=False)
                        for token_id in input_ids
                    ],
                    "lora_scalings": lora_scalings * steering_vector_norms[layer_idx],
                    "reward": reward,
                }
            )

        sample_lora_scalings = pad_sequences(
            sample_lora_scalings,
            pad_length=max(len(x) for x in sample_lora_scalings),
            pad_value=torch.nan,
        )
        sample_lora_scalings = sample_lora_scalings.float().cpu().numpy()

        sample_hidden_norms = torch.vstack(sample_hidden_norms).float().cpu().numpy()
        sample_cossims = pad_sequences(
            sample_cossims,
            pad_length=max(len(x) for x in sample_cossims),
            pad_value=torch.nan,
        )
        sample_cossims = sample_cossims.float().cpu().numpy()

        if sample_idx >= config.max_samples_to_plot:
            continue
        # save it
        save_single_lora_scaling(
            lora_scaling=sample_lora_scalings,
            savedir=os.path.join(
                config.output_dir, "lora_scalings_raw", f"sample_{sample_idx}"
            ),
        )
        scalings_hist_plots(
            lora_scalings=sample_lora_scalings / sample_hidden_norms,
            savepath=os.path.join(
                config.output_dir,
                "lora_scalings_hists_norm",
                f"sample_{sample_idx}.png",
            ),
        )
        scalings_hist_plots(
            lora_scalings=sample_lora_scalings,
            savepath=os.path.join(
                config.output_dir,
                "lora_scalings_hists",
                f"sample_{sample_idx}.png",
            ),
        )
        plot_scatters(
            X=sample_cossims,
            Y=sample_lora_scalings,
            savepath=os.path.join(
                config.output_dir,
                "lora_scalings_scatters",
                f"sample_{sample_idx}.png",
            ),
        )
        plot_scatters(
            X=sample_cossims,
            Y=sample_lora_scalings / sample_hidden_norms,
            savepath=os.path.join(
                config.output_dir,
                "lora_scalings_scatters_norm",
                f"sample_{sample_idx}.png",
            ),
        )

        system_w_prompt_len = len(
            tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": guidance_prompt},
                    {"role": "user", "content": ds["problem"][sample_idx]},
                ],
                add_generation_prompt=False,
                tokenize=True,
                padding=False,
                truncation=False,
                return_dict=True,
            )["input_ids"]
        )

        scalings_imshow_plot(
            lora_scalings=sample_lora_scalings,
            borders=[
                system_len,
                system_w_role_len,
                system_w_prompt_len,
                len(prompt_token_ids),
            ],
            savepath=os.path.join(
                config.output_dir,
                "lora_scalings_imshow",
                f"sample_{sample_idx}.png",
            ),
        )
        savetext(
            text=tokenizer.decode(input_ids, skip_special_tokens=False),
            savepath=os.path.join(
                config.output_dir,
                "lora_scalings_texts",
                f"sample_{sample_idx}.txt",
            ),
        )

        for layer_idx in range(num_layers):
            tokens = [
                tokenizer.decode(x, skip_special_tokens=False)
                for x in sample_tokens[layer_idx]
            ]
            scores = sample_lora_scalings[layer_idx] / sample_hidden_norms[layer_idx]
            scores = scores[: len(tokens)]
            assert scores.ndim == 1, scores.ndim

            token_scores(
                tokens=tokens,
                scores=scores,
                generation_offset=0,
                savepath=os.path.join(
                    config.output_dir,
                    "lora_scalings_token_scores",
                    f"sample_{sample_idx}",
                    f"layer_{layer_idx}.html",
                ),
            )

    def flip(x):
        if x.mean() < 0:
            return -x
        else:
            return x

    plot_line(
        [
            torch.mean(flip(lora_scalings) / torch.mean(hidden_norms))
            for lora_scalings, hidden_norms in zip(all_lora_scalings, all_hidden_norms)
        ],
        xlabel="Layer",
        ylabel="Mean Lora Scaling",
        savepath=os.path.join(config.output_dir, "global", "lora_scalings_mean.png"),
    )
    plot_line(
        [
            torch.median(flip(lora_scalings) / torch.mean(hidden_norms))
            for lora_scalings, hidden_norms in zip(all_lora_scalings, all_hidden_norms)
        ],
        xlabel="Layer",
        ylabel="Median Lora Scaling",
        savepath=os.path.join(config.output_dir, "global", "lora_scalings_median.png"),
    )
    plot_line(
        [
            torch.mean(torch.abs(lora_scalings) / torch.mean(hidden_norms))
            for lora_scalings, hidden_norms in zip(all_lora_scalings, all_hidden_norms)
        ],
        xlabel="Layer",
        ylabel="Median Lora Scaling",
        savepath=os.path.join(config.output_dir, "global", "lora_scalings_median.png"),
    )
    scalings_hist_plots(
        lora_scalings=all_lora_scalings,
        savepath=os.path.join(config.output_dir, "global", "scalings_hist.png"),
    )
    scalings_hist_plots(
        lora_scalings=[
            lora_scalings / torch.mean(hidden_norms)
            for lora_scalings, hidden_norms in zip(all_lora_scalings, all_hidden_norms)
        ],
        savepath=os.path.join(config.output_dir, "global", "scalings_hist_norm.png"),
    )

    scaled_for_json = []
    for row in for_json:
        row = copy.copy(row)
        row["lora_scalings"] /= all_hidden_norms[row["layer_idx"]].mean()
        row["lora_scalings"] = row["lora_scalings"].tolist()

        scaled_for_json.append(row)

    for_json = scaled_for_json

    save_token_acts(for_json, savepath=os.path.join(config.output_dir, "dump.json"))


if __name__ == "__main__":
    run()
