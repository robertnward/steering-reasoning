import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyrallis
import torch
from accelerate.utils import set_seed
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


@dataclass
class Config:
    seed: int
    model_path: str
    steering_vectors_path: str
    insert_token: str
    simple_prompt: bool
    with_system_prompt: bool
    only_lth_layer: Optional[int]

    scaling_max: int
    scaling_step: int
    savedir: str

    def __post_init__(self):
        self.savedir = os.path.join(
            self.savedir,
            f"only_lth_layer_{self.only_lth_layer}",
            f"inserted_token_{self.insert_token}",
            f"is_simple_prompt_{self.simple_prompt}",
            f"with_system_prompt_{self.with_system_prompt}",
        )
        os.makedirs(self.savedir, exist_ok=True)


def tokenize_prompt_without_system(
    tokenizer: PreTrainedTokenizer,
    conversation: List[Dict[str, str]],
    remove_system: bool,
    device: torch.device,
):
    tokenized_prompt = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=False,
        tokenize=True,
        padding=False,
        truncation=False,
        return_dict=True,
    )
    if remove_system:
        system_tokenized_prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            padding=False,
            truncation=False,
            return_dict=True,
        )
        decoded_prompt = tokenizer.decode(
            tokenized_prompt["input_ids"], skip_special_tokens=False
        )
        decoded_prompt_tokens = [
            tokenizer.decode(x, skip_special_tokens=False)
            for x in tokenized_prompt["input_ids"]
        ]
        logger.info(decoded_prompt)
        logger.info(f"{decoded_prompt_tokens}")
        system_decoded_prompt = tokenizer.decode(
            system_tokenized_prompt["input_ids"], skip_special_tokens=False
        )
        system_decoded_prompt_tokens = [
            tokenizer.decode(x, skip_special_tokens=False)
            for x in system_tokenized_prompt["input_ids"]
        ]
        logger.info(system_decoded_prompt)
        logger.info(f"{system_decoded_prompt_tokens}")

        # system + <|im_start|>\nuser<|im_end|>\n
        # -2 is for <|im_end|>\n
        offset = len(system_tokenized_prompt["input_ids"]) - 5

        tokenized_prompt = {
            "input_ids": tokenized_prompt["input_ids"][offset:-2],
            "attention_mask": tokenized_prompt["attention_mask"][offset:-2],
        }
    else:
        tokenized_prompt = {
            "input_ids": tokenized_prompt["input_ids"][:-2],
            "attention_mask": tokenized_prompt["attention_mask"][:-2],
        }
    tokenized_prompt = {
        "input_ids": torch.tensor(
            tokenized_prompt["input_ids"],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.tensor(
            tokenized_prompt["attention_mask"],
            dtype=torch.long,
            device=device,
        ),
    }
    return tokenized_prompt


def get_hook(steering_vector: torch.Tensor, positions: List[int]):
    def replace_residual_stream(module, inp, out):
        """
        module: the GPT2Block we're hooking
        inp:   tuple of inputs to the block (we ignore)
        out:   if output_attentions=False, this is a single tensor of shape
            (batch, seq_len, hidden_size). If it's a tuple, out[0] is that tensor.
        """
        # grab the tensor (either out itself or out[0] if tuple)
        if isinstance(out, tuple):
            h, rest = out[0], out[1:]
        else:
            h, rest = out, None

        # clone so we don't corrupt other computations
        h_mod = h.clone()
        assert len(positions) > 0, len(positions)
        for pos in positions:
            if pos < h.shape[1]:
                h_mod[:, pos, :] = steering_vector

        # return in the same format as we got it
        if rest is None:
            return h_mod
        else:
            return (h_mod, *rest)

    return replace_residual_stream


self_similarity = None
entropy = None


def get_self_similarity_hook(steering_vector: torch.Tensor):
    def self_similarity_hook(module, inp, out):
        global self_similarity

        # grab the tensor (either out itself or out[0] if tuple)
        if isinstance(out, tuple):
            h, _ = out[0], out[1:]
        else:
            h, _ = out, None

        if h.shape[1] != 1:
            assert self_similarity is None, self_similarity
            self_similarity = torch.nn.functional.cosine_similarity(
                steering_vector, h[:, -1]
            ).item()

        return out

    return self_similarity_hook


def get_entropy_hook(steering_vector: torch.Tensor):
    def entropy_hook(module, inp, out):
        global entropy

        # grab the tensor (either out itself or out[0] if tuple)
        if isinstance(out, tuple):
            h, _ = out[0], out[1:]
        else:
            h, _ = out, None

        if h.shape[1] == 1 and entropy is None:
            entropy = torch.distributions.Categorical(logits=h[0, 0]).entropy().item()

        return out

    return entropy_hook


@pyrallis.wrap()
def run(config: Config):
    set_seed(seed=config.seed, device_specific=False, deterministic=False)
    global self_similarity, entropy

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    steering_vectors = np.load(config.steering_vectors_path)
    steering_vectors = torch.from_numpy(steering_vectors).to(device)
    steering_vectors = steering_vectors / torch.linalg.norm(
        steering_vectors, dim=1, keepdim=True
    )

    if config.simple_prompt:
        conversation = [
            {"role": "user", "content": f'"{config.insert_token}"?'},
            {"role": "assistant", "content": f'"{config.insert_token}" "'},
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": f'What is the meaning of the word "{config.insert_token}"?',
            },
            {
                "role": "assistant",
                "content": f'The meaning of the word "{config.insert_token}" is "',
            },
        ]

    tokenized_prompt = tokenize_prompt_without_system(
        tokenizer=tokenizer,
        conversation=conversation,
        remove_system=not config.with_system_prompt,
        device=device,
    )
    positions = [
        i
        for i, a in enumerate(tokenized_prompt["input_ids"])
        if tokenizer.decode([a]) == "X"
    ]
    assert len(positions) > 0, [
        tokenizer.decode(x, skip_special_tokens=False)
        for x in tokenized_prompt["input_ids"]
    ]

    with open(os.path.join(config.savedir, "prompt.txt"), "+w") as f:
        f.write(
            tokenizer.decode(tokenized_prompt["input_ids"], skip_special_tokens=False)
        )

    for layer_idx, layer in tqdm(enumerate(model.model.layers), desc="Layers"):
        # Steering vector from any layer will be inserted into this layer
        if config.only_lth_layer is not None:
            layer = model.model.layers[config.only_lth_layer]

        layer_generations = []
        self_similarities = []
        entropies = []
        scalings = [1] + list(range(0, config.scaling_max, config.scaling_step))[1:]
        for scaling in tqdm(scalings, desc="Scaling"):
            self_similarity = None
            entropy = None
            steering_vector = scaling * steering_vectors[layer_idx]
            hook_handle = layer.register_forward_hook(
                get_hook(steering_vector=steering_vector, positions=positions)
            )
            self_similarity_hook_handle = model.model.layers[-1].register_forward_hook(
                get_self_similarity_hook(steering_vector=steering_vector)
            )
            entropy_hook_handle = model.get_output_embeddings().register_forward_hook(
                get_entropy_hook(steering_vector=steering_vector)
            )

            generated_ids = model.generate(
                input_ids=tokenized_prompt["input_ids"].unsqueeze(0),
                attention_mask=tokenized_prompt["attention_mask"].unsqueeze(0),
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )
            generated_ids = generated_ids.squeeze(0)

            decoded_generation = tokenizer.decode(
                generated_ids[len(tokenized_prompt["input_ids"]) :],
                skip_special_tokens=False,
            )
            layer_generations.append((scaling, decoded_generation))

            self_similarities.append(self_similarity)
            entropies.append(entropy)

            hook_handle.remove()
            self_similarity_hook_handle.remove()
            entropy_hook_handle.remove()

        with open(os.path.join(config.savedir, f"layer_{layer_idx}.txt"), "+w") as f:
            layer_generations = [
                "-" * 50 + f"\nScaling {scaling}.\n {decoded_generation}"
                for scaling, decoded_generation in layer_generations
            ]
            layer_generations_str = "\n\n".join(layer_generations)
            f.write(layer_generations_str)

        # Create primary axis
        fig, ax1 = plt.subplots()

        # Plot self-similarity on primary y-axis (left)
        ax1.plot(
            scalings, self_similarities, "o-", label="Self-Similarity", color="blue"
        )
        ax1.set_xlabel("Scaling")
        ax1.set_ylabel("Self-Similarity", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        # Create secondary axis for entropy
        ax2 = ax1.twinx()
        ax2.plot(scalings, entropies, "o-", label="Entropy", color="orange")
        ax2.set_ylabel("Entropy", color="orange")
        ax2.tick_params(axis="y", labelcolor="orange")

        # Title and grid
        plt.title("Scaling metrics")
        ax1.grid(True)

        # Save the figure
        plt.savefig(
            os.path.join(config.savedir, f"layer_{layer_idx}_scaling_metrics.png")
        )
        plt.close()


if __name__ == "__main__":
    run()
