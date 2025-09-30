import math
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
import vllm
from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from steering_reasoning.metrics.plots import (
    draw_pairwise_cossims,
    draw_pairwise_cossims_hist,
    get_logit_lens,
    save_logit_lens,
)
from steering_reasoning.steering_diff.config import Config
from steering_reasoning.train.rl.policy_model import (
    CustomCollator,
    subsample_ds,
    tokenize_example,
)
from steering_reasoning.train.rl.trainer import (
    input_ids_to_list,
    is_answer_present,
    pad_sequences,
)


class VLLMActor:
    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerBase):
        self.vllm_actor = vllm.LLM(
            model=config.model_path,
            trust_remote_code=True,
            seed=config.seed,
            enable_prefix_caching=False,
            enforce_eager=False,
            max_model_len=config.max_seq_length,
            max_seq_len_to_capture=config.max_seq_length * 2,
            dtype="bfloat16",
        )
        self.num_layers = len(
            self.vllm_actor.llm_engine.model_executor.driver_worker.model_runner.model.model.layers
        )
        self.zero_steering_vectors = [
            torch.zeros((1, 1, 1536)) for _ in range(self.num_layers)
        ]

        self.tokenizer = tokenizer

        stop_token_id = tokenizer.encode(config.stop_token, add_special_tokens=False)
        assert len(stop_token_id) == 1, (
            f"Stop token must be a single token, instead {stop_token_id}"
        )
        stop_token_id = stop_token_id[0]

        self.sampling_params = vllm.SamplingParams(
            n=config.num_generations_for_train,
            repetition_penalty=config.repetition_penalty,
            top_p=config.top_p,
            top_k=config.top_k,
            temperature=config.generation_temperature,
            stop_token_ids=[stop_token_id],
            seed=config.seed,
        )

        self.reset_steering_vectors()

    def reset_steering_vectors(self):
        update_steering_vectors(
            vllm_actor=self.vllm_actor,
            steering_vectors=self.zero_steering_vectors,
            update_layers=None,
            num_layers=self.num_layers,
        )

    def update_steering_vectors(
        self, steering_vectors: List[torch.Tensor], layer_idx: int
    ):
        self.reset_steering_vectors()

        update_steering_vectors(
            vllm_actor=self.vllm_actor,
            steering_vectors=steering_vectors,
            update_layers=[layer_idx],
            num_layers=self.num_layers,
        )

    def generate(self, prompts: Dict[str, List[int]]):
        return self.vllm_actor.generate(
            prompts, sampling_params=self.sampling_params, use_tqdm=False
        )


def get_generations_with_rewards(
    vllm_actor: VLLMActor,
    samples: List[Dict[str, torch.Tensor]],
    max_seq_length: int,
):
    for sample in samples:
        assert all(x == 1 for x in sample["prompt_attention_mask"]), sample[
            "prompt_attention_mask"
        ]
        assert all(x == 1 for x in sample["gt_cot_attention_mask"]), sample[
            "gt_cot_attention_mask"
        ]

    input_ids = [
        vllm.TokensPrompt(prompt_token_ids=sample["prompt_input_ids"])
        for sample in samples
    ]
    gt_cot_token_ids = [sample["gt_cot_input_ids"] for sample in samples]

    max_gen_tokens = [max_seq_length - len(x["prompt_token_ids"]) for x in input_ids]
    assert all(0 <= x < max_seq_length for x in max_gen_tokens), max_gen_tokens
    vllm_actor.sampling_params.max_tokens = max(max_gen_tokens)

    request_output: List[vllm.RequestOutput] = vllm_actor.generate(prompts=input_ids)

    generation_token_ids = []
    rewards = []
    for prompt_idx in range(len(samples)):
        generations = request_output[prompt_idx].outputs

        prompt_generation_token_ids = []
        prompt_rewards = []
        for generation in generations:
            sliced_generation_token_ids = list(
                generation.token_ids[: max_gen_tokens[prompt_idx]]
            )
            prompt_generation_token_ids.append(sliced_generation_token_ids)

            reward = int(
                is_answer_present(
                    processing_class=vllm_actor.tokenizer,
                    gt_cot_token_ids=gt_cot_token_ids[prompt_idx],
                    generation_token_ids=sliced_generation_token_ids,
                )[0]
            )
            prompt_rewards.append(reward)

        generation_token_ids.append(prompt_generation_token_ids)
        rewards.append(prompt_rewards)

    vllm_actor.sampling_params.seed += 1

    return generation_token_ids, rewards


def single_merge_inputs_and_generations(
    inputs: Dict[str, torch.Tensor], generation_token_ids: List[int]
):
    prompt_token_ids = input_ids_to_list(
        input_ids=inputs["prompt_input_ids"],
        attention_mask=inputs["prompt_attention_mask"],
    )

    input_ids = torch.hstack(
        [torch.tensor(prompt_token_ids), torch.tensor(generation_token_ids)]
    )
    attention_mask = torch.ones_like(input_ids)
    prompt_tokens_mask = torch.hstack(
        [torch.ones(len(prompt_token_ids)), torch.zeros(len(generation_token_ids))]
    )
    generation_tokens_mask = torch.hstack(
        [torch.zeros(len(prompt_token_ids)), torch.ones(len(generation_token_ids))]
    )

    return (
        input_ids.to(torch.long),
        attention_mask.to(torch.long),
        prompt_tokens_mask.to(torch.long),
        generation_tokens_mask.to(torch.long),
    )


def make_batch(
    sample: Dict[str, torch.Tensor],
    generation_token_ids: List[int],
    max_length: int,
    pad_token_id: int,
    device: torch.device,
):
    input_ids = []
    attention_mask = []
    prompt_tokens_mask = []
    generation_tokens_mask = []
    for gen_token_ids in generation_token_ids:
        (
            single_input_ids,
            single_attention_mask,
            single_prompt_tokens_mask,
            single_generation_tokens_mask,
        ) = single_merge_inputs_and_generations(
            inputs=sample, generation_token_ids=gen_token_ids
        )

        input_ids.append(single_input_ids)
        attention_mask.append(single_attention_mask)
        prompt_tokens_mask.append(single_prompt_tokens_mask)
        generation_tokens_mask.append(single_generation_tokens_mask)

    input_ids = pad_sequences(
        sequences=input_ids,
        pad_length=max_length,
        pad_value=pad_token_id,
    )
    attention_mask = pad_sequences(
        sequences=attention_mask, pad_length=max_length, pad_value=0
    )
    prompt_tokens_mask = pad_sequences(
        sequences=prompt_tokens_mask, pad_length=max_length, pad_value=0
    )
    generation_tokens_mask = pad_sequences(
        sequences=generation_tokens_mask, pad_length=max_length, pad_value=0
    )

    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    position_ids.masked_fill_(attention_mask.to(torch.bool) == 0, 0)

    return (
        input_ids.to(device=device),
        attention_mask.to(device=device),
        prompt_tokens_mask.to(device=device),
        generation_tokens_mask.to(device=device),
        position_ids.to(device=device),
    )


def get_steering_vectors_svm(
    all_hiddens: np.ndarray,
    tokens_mask: np.ndarray,
    rewards: List[int],
):
    num_layers = len(all_hiddens[0])
    pos_mask = np.array(rewards, dtype=int)
    steering_vectors = []
    layer_f1s = []
    for layer_idx in range(num_layers):
        hiddens = all_hiddens[:, layer_idx]
        hiddens = [x[mask.astype(bool)] for x, mask in zip(hiddens, tokens_mask)]
        hiddens_mean = np.vstack([x.mean(0) for x in hiddens])

        svm = SVC(kernel="linear")
        svm.fit(hiddens_mean, pos_mask)
        predictions = svm.predict(hiddens_mean)
        f1 = f1_score(y_true=pos_mask, y_pred=predictions)
        layer_f1s.append(f1)

        steering_vector = svm.coef_.squeeze(0).copy()
        steering_vector /= np.linalg.norm(svm.coef_)

        steering_vectors.append(steering_vector)

    layer_f1s = np.array(layer_f1s)

    return steering_vectors, {"f1": layer_f1s}


def get_steering_vectors_diff(
    all_hiddens: np.ndarray,
    tokens_mask: np.ndarray,
    rewards: List[int],
):
    num_layers = len(all_hiddens[0])
    pos_mask = np.array(rewards, dtype=bool)
    steering_vectors = []
    for layer_idx in range(num_layers):
        hiddens = all_hiddens[:, layer_idx]
        hiddens = [x[mask.astype(bool)] for x, mask in zip(hiddens, tokens_mask)]

        pos_hiddens = [x for x, pm in zip(hiddens, pos_mask) if pm]
        neg_hiddens = [x for x, pm in zip(hiddens, pos_mask) if not pm]

        pos_hiddens_mean = np.vstack([x.mean(0) for x in pos_hiddens]).mean(0)
        neg_hiddens_mean = np.vstack([x.mean(0) for x in neg_hiddens]).mean(0)

        steering_vector = pos_hiddens_mean - neg_hiddens_mean
        steering_vector /= np.linalg.norm(steering_vector)

        steering_vectors.append(steering_vector)

    return steering_vectors, {}


def get_steering_vectors(
    config: Config,
    all_hiddens: np.ndarray,
    tokens_mask: torch.Tensor,
    rewards: List[int],
):
    # don't consider the hidden states after an embedding layer
    if config.calc_type == "diff":
        steering_vectors, metrics = get_steering_vectors_diff(
            all_hiddens=all_hiddens,
            tokens_mask=tokens_mask,
            rewards=rewards,
        )
    elif config.calc_type == "svm":
        steering_vectors, metrics = get_steering_vectors_svm(
            all_hiddens=all_hiddens,
            tokens_mask=tokens_mask,
            rewards=rewards,
        )
    else:
        raise NotImplementedError

    steering_vectors = [torch.from_numpy(x) for x in steering_vectors]
    assert all(len(x) == config.hidden_dim for x in steering_vectors), [
        len(x) == config.hidden_dim for x in steering_vectors
    ]

    return steering_vectors, metrics


def update_steering_vectors(
    vllm_actor: vllm.LLM,
    steering_vectors: List[torch.Tensor],
    update_layers: Optional[List[int]],
    num_layers: int,
):
    if steering_vectors[0].dim() == 1:
        steering_vectors = [x.unsqueeze(0).unsqueeze(0) for x in steering_vectors]

    assert len(steering_vectors) == num_layers, (
        len(steering_vectors),
        num_layers,
    )

    if update_layers is None:
        update_layers = list(range(num_layers))

    for layer_idx in update_layers:
        vllm_actor.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
            weights=[
                (
                    f"model.layers.{layer_idx}.steering_vector.steering_vector",
                    steering_vectors[layer_idx],
                )
            ]
        )


def cossims_plots(steering_vectors: List[List[np.ndarray]], savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    fig, axes = plt.subplots(4, 7, figsize=(18, 9), sharex=True, sharey=True)
    fig.suptitle("Cossims of vectors between samples", fontsize=16, y=0.94)

    for layer_idx, ax in enumerate(axes.ravel()):
        if layer_idx >= len(steering_vectors[0]):
            break

        vecs_at_layer = [x[layer_idx] for x in steering_vectors]

        V = np.vstack(vecs_at_layer)

        # L2‑normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("At least one vector has zero norm.")
        Vn = V / norms

        # Cosine similarity matrix
        sim_matrix = Vn @ Vn.T  # (N, N)
        mask = np.triu(np.ones(sim_matrix.shape), k=1).astype(bool)
        sims = sim_matrix[mask]

        ax.hist(sims, bins=50)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Cossim", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    plt.close()


def cossims_imshow_plots(steering_vectors: List[List[np.ndarray]], savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    fig, axes = plt.subplots(4, 7, figsize=(36, 18), sharex=True, sharey=True)
    fig.suptitle("Cossims of vectors between samples", fontsize=16, y=0.94)

    for layer_idx, ax in enumerate(axes.ravel()):
        if layer_idx >= len(steering_vectors[0]):
            break

        vecs_at_layer = [x[layer_idx] for x in steering_vectors]

        V = np.vstack(vecs_at_layer)

        # L2‑normalize rows
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("At least one vector has zero norm.")
        Vn = V / norms

        # Cosine similarity matrix
        sim_matrix = Vn @ Vn.T  # (N, N)

        im = ax.imshow(sim_matrix, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Vector idx", fontsize=9)
        ax.set_ylabel("Vector idx", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    fig.colorbar(im, cax=cbar_ax)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    plt.close()


def layer_alpha_plots(config: Config, scores: List[List[float]], savepath: str):
    xs = np.linspace(config.alpha_min, config.alpha_max, config.alpha_num_steps)

    fig, axes = plt.subplots(4, 7, figsize=(18, 9), sharex=True, sharey=True)
    fig.suptitle("Reward vs. alpha at different layers", fontsize=16, y=0.94)

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(scores):
            break
        ax.plot(xs, scores[idx])
        ax.set_title(f"Layer {idx}", fontsize=9)
        ax.set_xlabel("alpha", fontsize=9)
        ax.set_ylabel("Reward", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    plt.close()


def decode_generations(
    tokenizer: PreTrainedTokenizerBase, generation_token_ids: List[List[int]]
):
    decoded_generations = []
    for sample_idx in range(len(generation_token_ids)):
        accum = []
        for generation_idx in range(len(generation_token_ids[sample_idx])):
            decoded_generation = tokenizer.decode(
                generation_token_ids[sample_idx][generation_idx],
                skip_special_tokens=False,
            )
            accum.append(decoded_generation)
        decoded_generations.append(accum)

    return decoded_generations


def get_layer_rewards(
    config: Config,
    samples: Dict[str, torch.Tensor],
    vllm_actor: VLLMActor,
    steering_vectors: List[np.ndarray],
):
    layer_rewards = []
    all_decoded_generations = []
    vllm_actor.sampling_params.n = config.num_generations_for_test
    for layer_idx in trange(vllm_actor.num_layers, desc="Layers"):
        alpha_rewards = []
        gens_accum = []
        for alpha in tqdm(
            np.linspace(config.alpha_min, config.alpha_max, config.alpha_num_steps),
            desc="Alphas",
        ):
            vllm_actor.update_steering_vectors(
                steering_vectors=[x * alpha for x in steering_vectors],
                layer_idx=layer_idx,
            )

            generation_token_ids, rewards = get_generations_with_rewards(
                vllm_actor=vllm_actor,
                samples=samples,
                max_seq_length=config.max_seq_length,
            )

            alpha_rewards.append(np.mean(rewards, axis=1))

            # Save the decoded generations for a later printing
            decoded_generations = decode_generations(
                tokenizer=vllm_actor.tokenizer,
                generation_token_ids=generation_token_ids,
            )
            gens_accum.append(((layer_idx, alpha), (decoded_generations, rewards)))

        layer_rewards.append(alpha_rewards)
        all_decoded_generations.append(gens_accum)

    layer_rewards = np.array(layer_rewards)
    # put the `samples` dimention first
    layer_rewards = layer_rewards.transpose(2, 0, 1)

    return layer_rewards, all_decoded_generations


def get_ds_layer_rewards(
    config: Config,
    ds: Dataset,
    vllm_actor: VLLMActor,
    steering_vectors: List[np.ndarray],
):
    all_rewards = []
    for _, samples in samples_generator(
        ds=ds, batch_size=min(len(ds), config.test_gen_batch_size), desc="Test Samples"
    ):
        layer_rewards, _ = get_layer_rewards(
            config=config,
            samples=samples,
            vllm_actor=vllm_actor,
            steering_vectors=steering_vectors,
        )

        all_rewards.append(layer_rewards)

    all_rewards = np.vstack(all_rewards)
    ds_layer_rewards = np.mean(all_rewards, axis=0)

    return ds_layer_rewards


def save_steering_vectors(config: Config, all_steering_vectors: List[List[np.ndarray]]):
    savepath = os.path.join(config.savedir, "steering_vectors", "steering_vectors.pkl")
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    with open(savepath, "+wb") as f:
        pickle.dump(all_steering_vectors, f)


def plot_steering_vectors(
    config: Config,
    rng: np.random.Generator,
    all_steering_vectors: List[List[np.ndarray]],
    num_layers: int,
):
    for layer_idx in trange(num_layers, desc="Plot Steering Vectors. Layer"):
        vecs_at_layer = [x[layer_idx] for x in all_steering_vectors]

        savedir = os.path.join(
            config.savedir, "plots", "steering_vectors", f"layer_{layer_idx}"
        )
        os.makedirs(savedir, exist_ok=True)

        if len(vecs_at_layer) > 50:
            indices = list(range(len(vecs_at_layer)))
            rng.choice(indices, size=50, replace=False)
            subsampled_vecs_at_layer = [vecs_at_layer[i] for i in indices]

            draw_pairwise_cossims(
                steering_vectors=subsampled_vecs_at_layer, savedir=savedir
            )
        else:
            draw_pairwise_cossims(steering_vectors=vecs_at_layer, savedir=savedir)

        draw_pairwise_cossims_hist(steering_vectors=vecs_at_layer, savedir=savedir)


def draw_f1(values: np.ndarray, savepath: str):
    plt.figure(figsize=(10, 10))
    plt.plot(values)
    plt.xlabel("Layer")
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def plot_steering_metrics(metrics: Dict[str, np.ndarray], savedir: str):
    os.makedirs(savedir, exist_ok=True)
    for metric_name, metric_values in metrics.items():
        savepath = os.path.join(savedir, f"{metric_name}.png")

        if metric_name == "f1":
            draw_f1(metric_values, savepath)
        else:
            raise NotImplementedError


def get_model_and_tokenizer(config: Config):
    model = (
        AutoModelForCausalLM.from_pretrained(
            config.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        .to(config.device)
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
        truncation_side="right",
        model_max_length=config.max_seq_length,
    )

    return model, tokenizer


def get_ds(
    config: Config, rng: np.random.Generator, tokenizer: PreTrainedTokenizerBase
):
    # Dataset Preparation
    ds = load_dataset(config.dataset_path)
    ds["train"] = ds["train"].filter(lambda x: x["source"] in config.train_datasets)
    ds["test"] = ds["test"].filter(lambda x: x["source"] in config.test_datasets)
    ds["train"] = ds["train"].add_column("index", list(range(len(ds["train"]))))
    ds["test"] = ds["test"].add_column("index", list(range(len(ds["test"]))))

    ds = ds.map(
        tokenize_example(
            tokenizer=tokenizer,
            stop_token=config.stop_token,
            max_seq_length=config.max_seq_length,
        ),
        batched=False,
        num_proc=None,
        # remove_columns=["prompt", "answer"],
    )
    ds = ds.filter(
        lambda x: (len(x["prompt_attention_mask"]) + len(x["gt_cot_attention_mask"]))
        <= config.max_seq_length
    )
    ds = ds.shuffle(seed=config.seed)

    if config.num_train_samples is not None:
        ds["train"] = concatenate_datasets(
            [
                subsample_ds(
                    ds["train"].filter(lambda x: x["source"] == dataset_source),
                    num_samples=config.num_train_samples,
                    np_rng=rng,
                )
                for dataset_source in config.train_datasets
            ]
        )

    if config.num_test_samples is not None:
        ds["test"] = concatenate_datasets(
            [
                subsample_ds(
                    ds["test"].filter(lambda x: x["source"] == dataset_source),
                    num_samples=config.num_test_samples,
                    np_rng=rng,
                )
                for dataset_source in config.test_datasets
            ]
        )

    return ds


@torch.no_grad()
def get_hidden_states(
    config: Config,
    model: AutoModelForCausalLM,
    sample: Dict[str, torch.Tensor],
    gen_token_ids: List[List[int]],
    pad_token_id: int,
):
    all_hiddens = None
    all_attention_masks = None
    for offset in range(0, len(gen_token_ids), config.batch_size):
        (
            input_ids,
            attention_mask,
            prompt_tokens_mask,
            generation_tokens_mask,
            position_ids,
        ) = make_batch(
            sample=sample,
            generation_token_ids=gen_token_ids[offset : offset + config.batch_size],
            max_length=config.max_seq_length,
            pad_token_id=pad_token_id,
            device=config.device,
        )

        hidden_states = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        ).hidden_states

        # ignore the hidden states after embedding layer
        # there is no steering vector at this place
        hidden_states = hidden_states[1:]

        if all_hiddens is None:
            all_hiddens = [x.cpu().float().numpy() for x in hidden_states]
            assert all_attention_masks is None
            all_attention_masks = attention_mask.cpu().numpy()
        else:
            all_hiddens = [
                np.vstack((x, y.cpu().float().numpy()))
                for x, y in zip(all_hiddens, hidden_states)
            ]
            assert all_attention_masks is not None
            all_attention_masks = np.vstack(
                [all_attention_masks, attention_mask.cpu().numpy()]
            )

    # [samples, layers, tokens, hiddens]
    all_hiddens = np.vstack([np.expand_dims(x, 0) for x in all_hiddens]).transpose(
        1, 0, 2, 3
    )
    assert (
        all_hiddens.shape[0] == all_attention_masks.shape[0]
        and all_hiddens.shape[2] == all_attention_masks.shape[1]
    ), (all_hiddens.shape, all_attention_masks.shape)

    return all_hiddens, all_attention_masks


def average_steering_vectors(all_steering_vectors: List[List[torch.Tensor]]):
    num_layers = len(all_steering_vectors[0])
    logger.info(f"Num steering vectors: {len(all_steering_vectors)}")
    averaged_steering_vectors = []
    for layer_idx in trange(num_layers, desc="Average Vectors. Layers"):
        vecs_at_layer = [x[layer_idx] for x in all_steering_vectors]
        averaged_steering_vector = torch.mean(torch.vstack(vecs_at_layer), dim=0)
        averaged_steering_vector /= torch.linalg.norm(averaged_steering_vector)
        averaged_steering_vectors.append(averaged_steering_vector)

    return averaged_steering_vectors


def plot_steering_vectors_norms(steering_vectors: List[torch.Tensor], savedir: str):
    savepath = os.path.join(
        savedir, "plots", "steering_vectors", "averaged_vectors_norms.png"
    )
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    norms = [torch.linalg.norm(x).round(decimals=2) for x in steering_vectors]
    plt.figure(figsize=(10, 10))
    plt.plot(norms)
    plt.xlabel("Layer")
    plt.ylabel("Norm")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def save_decoded_generations(
    decoded_generations: List[
        List[Tuple[Tuple[int, float], Tuple[List[List[str]], List[List[int]]]]]
    ],
    savedir: str,
):
    for layer_idx in range(len(decoded_generations)):
        for alpha_idx in range(len(decoded_generations[layer_idx])):
            (layer, alpha), (generations, rewards) = decoded_generations[layer_idx][
                alpha_idx
            ]
            assert len(generations) == 1, len(generations)
            assert len(rewards) == 1, len(rewards)
            generations = generations[0]
            rewards = rewards[0]

            savepath = os.path.join(
                savedir,
                "generations",
                f"layer_{layer}",
                f"alpha_{alpha}",
                "generations.txt",
            )
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            with open(savepath, "+w") as f:
                for idx, (generation, reward) in enumerate(zip(generations, rewards)):
                    f.write(f"\nGeneration {idx}")
                    f.write(f"\nReward: {reward}")
                    f.write(f"\nGeneration: {generation}")
                    f.write("\n\n")


def samples_generator(ds: Dataset, batch_size: int, desc: str):
    accum_idx = []
    accum = []

    for sample_idx, sample in tqdm(enumerate(ds), desc=desc, total=len(ds)):
        accum_idx.append(sample_idx)
        accum.append(sample)

        if len(accum) == batch_size:
            yield accum_idx, accum
            accum_idx = []
            accum = []


def plot_rewards(all_rewards: List[List[int]], savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    max_gens = len(all_rewards[0])
    powers_of_two = [2**x for x in range(int(math.log2(max_gens)))] + [max_gens]
    num_rows = 2
    num_cols = math.ceil(len(powers_of_two) / num_rows)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(18, 9), sharex=True, sharey=True
    )
    fig.suptitle("Rewards for generations", fontsize=16, y=0.94)

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(powers_of_two):
            break
        mean_rewards = [np.mean(x[: powers_of_two[idx]]) for x in all_rewards]

        ax.hist(mean_rewards, bins=50, range=(0, 1))
        ax.set_xlim(0, 1)
        ax.set_title(f"@{powers_of_two[idx]}", fontsize=9)
        ax.set_xlabel("Mean Reward", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    plt.close()


def plot_num_pos(all_rewards: List[List[int]], savepath: str):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    max_gens = len(all_rewards[0])
    powers_of_two = [2**x for x in range(int(math.log2(max_gens)))] + [max_gens]
    num_rows = 2
    num_cols = math.ceil(len(powers_of_two) / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 9), sharey=True)
    fig.suptitle("Positive generations", fontsize=16, y=0.94)

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(powers_of_two):
            break
        num_pos = [np.sum(x[: powers_of_two[idx]]) for x in all_rewards]

        ax.hist(num_pos, bins=50)
        ax.set_xlim(0, powers_of_two[idx])
        ax.set_title(f"@{powers_of_two[idx]}", fontsize=9)
        ax.set_xlabel("Num Positives", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    plt.close()


def check_consistency(
    config: Config,
    rng: np.random.Generator,
    all_hiddens: List[torch.Tensor],
    tokens_mask: torch.Tensor,
    rewards: List[int],
    sample_idx: int,
):
    indices = np.arange(len(rewards))

    all_steering_vectors = []
    for _ in range(config.num_reiterations_for_check_consistency):
        sub_indices = rng.choice(
            indices, size=config.check_consistency_sample_size, replace=False
        )
        sub_hiddens = np.array([all_hiddens[idx] for idx in sub_indices])
        sub_rewards = np.array([rewards[idx] for idx in sub_indices])
        sub_tokens_mask = np.array([tokens_mask[idx] for idx in sub_indices])
        if len(set(sub_rewards)) == 1:
            continue

        steering_vectors, _ = get_steering_vectors(
            config=config,
            all_hiddens=sub_hiddens,
            tokens_mask=sub_tokens_mask,
            rewards=sub_rewards,
        )
        all_steering_vectors.append(steering_vectors)

    cossims_plots(
        steering_vectors=all_steering_vectors,
        savepath=os.path.join(
            config.savedir,
            "plots",
            "steering_vectors",
            "check_consistency",
            f"sample_{sample_idx}_rew_{np.mean(rewards)}_cossims.png",
        ),
    )


def plot_averaged_steering_vector_change(
    all_steering_vectors: List[torch.Tensor], savepath: str
):
    cum_averaged_steering_vectors = []
    for idx in trange(
        1,
        len(all_steering_vectors),
        desc="Averaged Steering Vector Change. Num Vectors",
    ):
        averaged_steering_vector = average_steering_vectors(
            all_steering_vectors=all_steering_vectors[:idx]
        )
        cum_averaged_steering_vectors.append(averaged_steering_vector)

    cossims_imshow_plots(
        steering_vectors=cum_averaged_steering_vectors, savepath=savepath
    )


@pyrallis.wrap()
def run(config: Config):
    with torch.no_grad():
        rng = np.random.default_rng(seed=config.seed)
        model, tokenizer = get_model_and_tokenizer(config)

        config.hidden_dim = model.get_input_embeddings().weight.shape[1]

        vllm_actor = VLLMActor(config=config, tokenizer=tokenizer)

        ds = get_ds(config=config, rng=rng, tokenizer=tokenizer)

        data_collator = CustomCollator(tokenizer=tokenizer)

        all_steering_vectors = []
        all_steering_metrics = defaultdict(list)
        all_rewards = []

        num_dropped = 0

        for sample_ids, samples in samples_generator(
            ds=ds["train"], batch_size=config.train_gen_batch_size, desc="Samples"
        ):
            vllm_actor.reset_steering_vectors()

            vllm_actor.sampling_params.n = config.num_generations_for_train
            generation_token_ids, rewards = get_generations_with_rewards(
                vllm_actor=vllm_actor,
                samples=samples,
                max_seq_length=config.max_seq_length,
            )

            for sample_idx, sample, gen_token_ids, rews in zip(
                sample_ids, samples, generation_token_ids, rewards
            ):
                all_rewards.append(rews)
                # if there is only one class
                if len(set(rews)) == 1:
                    num_dropped += 1
                    continue

                all_hiddens, attention_mask = get_hidden_states(
                    config=config,
                    model=model,
                    sample=data_collator([sample]),
                    gen_token_ids=gen_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )

                steering_vectors, metrics = get_steering_vectors(
                    config=config,
                    all_hiddens=all_hiddens,
                    tokens_mask=attention_mask,
                    rewards=rews,
                )
                all_steering_vectors.append(steering_vectors)
                for metric_key, metric_values in metrics.items():
                    all_steering_metrics[metric_key].append(metric_values)

                if config.check_consistency:
                    check_consistency(
                        config=config,
                        rng=rng,
                        all_hiddens=all_hiddens,
                        tokens_mask=attention_mask,
                        rewards=rews,
                        sample_idx=sample_idx,
                    )

                if config.plot_sample_specific_results:
                    layer_rewards, decoded_generations = get_layer_rewards(
                        config=config,
                        samples=samples,
                        vllm_actor=vllm_actor,
                        steering_vectors=steering_vectors,
                    )
                    save_decoded_generations(
                        decoded_generations=decoded_generations, savedir=config.savedir
                    )
                    assert layer_rewards.shape[0] == 1, layer_rewards.shape
                    layer_rewards = layer_rewards[0]
                    layer_alpha_plots(
                        config=config,
                        scores=layer_rewards,
                        savepath=os.path.join(
                            config.savedir,
                            "plots",
                            config.calc_type,
                            "layer-alpha",
                            f"sample_{sample_idx}.png",
                        ),
                    )

                    plot_steering_metrics(
                        metrics=metrics,
                        savedir=os.path.join(
                            config.savedir,
                            "plots",
                            "steering_metrics",
                            f"sample_{sample_idx}",
                        ),
                    )

            if config.debug_on_single_sample:
                break

    logger.info(f"Dropped {num_dropped} samples.")

    # Plot dataset-wise metrics
    for metric_key, metric_values in all_steering_metrics.items():
        all_steering_metrics[metric_key] = np.vstack(metric_values).mean(0)
    plot_steering_metrics(
        metrics=metrics,
        savedir=os.path.join(
            config.savedir,
            "plots",
            "steering_metrics",
            f"ds_of_{config.num_train_samples}_samples",
        ),
    )

    save_steering_vectors(config=config, all_steering_vectors=all_steering_vectors)
    cossims_plots(
        steering_vectors=all_steering_vectors,
        savepath=os.path.join(
            config.savedir, "plots", "steering_vectors", "cossims.png"
        ),
    )

    plot_rewards(
        all_rewards=all_rewards,
        savepath=os.path.join(config.savedir, "plots", "rewards.png"),
    )
    plot_num_pos(
        all_rewards=all_rewards,
        savepath=os.path.join(config.savedir, "plots", "num_pos.png"),
    )

    averaged_steering_vectors = average_steering_vectors(
        all_steering_vectors=all_steering_vectors
    )
    plot_steering_vectors_norms(
        steering_vectors=averaged_steering_vectors, savedir=config.savedir
    )
    plot_averaged_steering_vector_change(
        all_steering_vectors=all_steering_vectors,
        savepath=os.path.join(
            config.savedir,
            "plots",
            "steering_vectors",
            "averaged_steering_vector_change.png",
        ),
    )

    top_tokens_with_scores = get_logit_lens(
        ranking_type="cossim",
        model=model,
        steering_vectors=[x.cpu().numpy() for x in steering_vectors],
        tokenizer=tokenizer,
    )
    save_logit_lens(
        top_tokens_with_scores=top_tokens_with_scores,
        savepath=os.path.join(config.savedir, "logit_lens_cossim.txt"),
    )

    # eval on both splits
    # for split in ["train", "test"]:
    #     ds_layer_rewards = get_ds_layer_rewards(
    #         config=config,
    #         ds=ds[split],
    #         vllm_actor=vllm_actor,
    #         steering_vectors=averaged_steering_vectors,
    #     )

    #     layer_alpha_plots(
    #         config=config,
    #         scores=ds_layer_rewards,
    #         savepath=os.path.join(
    #             config.savedir,
    #             "plots",
    #             config.calc_type,
    #             "layer-alpha",
    #             f"{split}-ds.png",
    #         ),
    #     )


if __name__ == "__main__":
    run()
