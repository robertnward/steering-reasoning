"""
Tuned Lens (single-layer) trainer with Hugging Face + Accelerate + Pyrallis + Weights & Biases.

Changes in this version
-----------------------
- **Exactly 1 epoch** of training over the train split.
- **Eval runs once per epoch** (i.e., after the epoch completes) over the full eval split.
- **Saves projection weights only at the very end** (no intermediate checkpoints).
- Reuses a single `hook_fn(cache)` for both train and eval.
"""

import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Dict

import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, load_from_disk
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


# -------------------------
# Config (parsed by pyrallis)
# -------------------------
@dataclass
class Config:
    seed: int

    # Model & tokenizer
    model_path: str
    dataset_path: str

    # Which layer to hook
    layer_idx: int

    # Training
    per_device_batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: float
    mixed_precision: str
    clip_grad_norm: float

    # Logging
    log_every_steps: int

    # Saving
    savedir: str

    # Prompt formatting for tokenize_example
    stop_token: str
    max_seq_length: int
    template_type: str

    dataset_type: str

    # wandb
    project: str
    entity: str
    group: str
    job_type: str
    name: str

    def __post_init__(self):
        model_name = os.path.basename(self.model_path)
        if "deepscaler" in self.dataset_path:
            dataset = self.dataset_type
        else:
            dataset = self.dataset_path

        self.group = os.path.join(model_name, dataset)
        self.job_type = f"tuned_lens_layer-{self.layer_idx}"
        self.name = f"lr-{self.learning_rate}"

        self.savedir = os.path.join(self.savedir, self.group, self.job_type, self.name)


# -------------------------
# Small utils
# -------------------------


def hook_fn(cache: Dict[str, torch.Tensor]):
    """Return a forward hook that writes the module output to `cache['hidden']`."""

    def _fn(_m, _i, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        cache["hidden"] = h

    return _fn


class TunedLensProjection(nn.Module):
    """Simple linear projection from hidden size -> vocab size."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)  # [B, T, V]


# -------------------------
# Data pipeline
# -------------------------


def make_loader(
    ds,
    config: Config,
    accelerator: Accelerator,
    shuffle: bool,
    collator: DataCollatorForLanguageModeling,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=config.per_device_batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=collator,
    )


# -------------------------
# Train / Eval
# -------------------------


def kl_loss_from_logits(
    logits_lens: torch.Tensor, logits_teacher: torch.Tensor, attn_mask: torch.Tensor
) -> torch.Tensor:
    """Token-weighted KL(p_teacher || p_lens)."""
    log_p_lens = logits_lens.log_softmax(dim=-1).to(torch.float64)
    p_teacher = logits_teacher.softmax(dim=-1).to(torch.float64)
    per_tok_kl = F.kl_div(log_p_lens, p_teacher, reduction="none").sum(dim=-1)
    mask = attn_mask.to(torch.float64)
    loss = (per_tok_kl * mask).sum() / (mask.sum() + 1e-8)
    return loss


@torch.no_grad()
def evaluate_kl(
    model,
    proj: TunedLensProjection,
    dataloader,
    accelerator: Accelerator,
    layer_idx: int,
) -> float:
    model.eval()
    proj.eval()
    total_kl = 0.0
    total_tokens = 0.0

    local_cache: Dict[str, torch.Tensor] = {"hidden": None}
    base = accelerator.unwrap_model(model)
    layer = base.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn(local_cache))

    for batch in dataloader:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        out = model(**batch)
        hidden = local_cache.get("hidden")
        if hidden is None:
            raise RuntimeError("Hook didn't capture activations. Check `layer_idx`.")
        logits_teacher = out.logits

        hidden_proj = proj(hidden)
        logit_lens = do_logit_lens(model=base, hidden=hidden_proj)

        kl = kl_loss_from_logits(logit_lens, logits_teacher, batch["attention_mask"])
        kl_val = accelerator.gather_for_metrics(kl.detach()).mean().item()
        ntoks = (
            accelerator.gather_for_metrics(batch["attention_mask"].float().sum())
            .sum()
            .item()
        )
        total_kl += kl_val * ntoks
        total_tokens += ntoks

    proj.train()
    handle.remove()
    return float(total_kl / max(total_tokens, 1.0))


def save_projection(
    accelerator: Accelerator, lens: TunedLensProjection, config: Config, layer_idx: int
):
    """Save ONLY the projection weights/bias (no metadata)."""
    if accelerator.is_main_process:
        os.makedirs(config.savedir, exist_ok=True)
        path = os.path.join(config.savedir, f"projection_layer{layer_idx}.pt")
        # {"proj.weight", "proj.bias"}
        state = accelerator.unwrap_model(lens).state_dict()
        torch.save(state, path)
        accelerator.print(f"[Saved weights] {path}")


def no_autocast(accelerator: Accelerator):
    dev = accelerator.device.type
    if dev == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    try:
        return torch.autocast(dev, enabled=False)
    except Exception:
        return nullcontext()


def print_example(
    accelerator: Accelerator,
    example: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
):
    if accelerator.is_main_process:
        accelerator.print("Example", example)
        accelerator.print(
            "input_ids",
            [
                tokenizer.decode(x, skip_special_tokens=False)
                for x in example["input_ids"]
            ],
        )


def do_logit_lens(model, hidden):
    unembed = model.get_output_embeddings()
    norm = model.model.norm
    dtype = unembed.weight.dtype

    return unembed(norm(hidden.to(dtype)))


def calc_mean_prob(logit_lens):
    probs = logit_lens.float().softmax(dim=-1)
    mean_prob = probs.squeeze(0).mean(dim=0)
    return mean_prob


# -------------------------
# Main
# -------------------------
@pyrallis.wrap()
def main(config: Config):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.grad_accum_steps,
        log_with="wandb",
    )
    set_seed(config.seed, device_specific=False, deterministic=False)

    # ----- wandb via accelerate -----
    tracker_kwargs = {
        "entity": config.entity,
        "group": config.group,
        "job_type": config.job_type,
        "name": config.name,
    }
    accelerator.init_trackers(
        project_name=config.project,
        config=asdict(config),
        init_kwargs={"wandb": tracker_kwargs},
    )

    accelerator.print(config)

    # --- Model & tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, use_fast=True, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    # Freeze base model parameters; train lens only
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # --- Data ---
    ds = load_from_disk(config.dataset_path)

    ds = ds if isinstance(ds, DatasetDict) else DatasetDict({"train": ds})

    def merge(x):
        return {
            "input_ids": torch.tensor(x["prompt_tokens"] + x["generation_tokens"]),
            "attention_mask": torch.ones(
                len(x["prompt_tokens"] + x["generation_tokens"])
            ),
        }

    ds = ds.map(merge)
    ds = ds.remove_columns(
        ["prompt_idx", "generation_idx", "prompt", "generation", "reward"]
    )
    print_example(accelerator, ds["train"][0], tokenizer)

    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id,
        completion_only_loss=False,
        return_tensors="pt",
    )

    train_loader = make_loader(
        ds["train"], config, accelerator, shuffle=True, collator=collator
    )
    eval_ds = (
        ds["validation"]
        if "validation" in ds
        else ds["train"].select(range(min(256, len(ds["train"]))))
    )
    eval_loader = make_loader(
        eval_ds, config, accelerator, shuffle=False, collator=collator
    )

    proj = TunedLensProjection(hidden_size=model.config.hidden_size).to(model.dtype)

    # Optim + scheduler (total steps = optimizer steps in 1 epoch)
    opt = torch.optim.AdamW(
        proj.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # number of optimizer updates per epoch = ceil(num_batches / grad_accum_steps)
    updates_per_epoch = math.ceil(
        len(train_loader)
        / (max(1, config.grad_accum_steps) * accelerator.num_processes)
    )
    total_steps = updates_per_epoch  # exactly 1 epoch
    if isinstance(config.warmup_steps, float) and 0 < config.warmup_steps < 1:
        warmup = config.warmup_steps * total_steps
    else:
        warmup = min(config.warmup_steps, total_steps)
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=warmup * accelerator.num_processes,
        num_training_steps=total_steps * accelerator.num_processes,
    )

    # Prepare distributed
    model, proj, opt, sched, train_loader, eval_loader = accelerator.prepare(
        model, proj, opt, sched, train_loader, eval_loader
    )

    # Register train hook
    act_cache: Dict[str, torch.Tensor] = {"hidden": None}
    base = accelerator.unwrap_model(model)
    layer = base.model.layers[config.layer_idx]
    handle = layer.register_forward_hook(hook_fn(act_cache))
    accelerator.print(f"[hook] Attached to base.model.layers[{config.layer_idx}]")

    # --- Training: exactly 1 epoch ---
    global_step = 0
    model.eval()
    proj.train()

    accum_mean_prob = []
    accum_mean_prob_proj = []
    accum_mean_prob_true = []

    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with accelerator.accumulate(proj):
            act_cache["hidden"] = None
            with torch.no_grad():
                out = model(**batch)
            hidden = act_cache.get("hidden")
            if hidden is None:
                raise RuntimeError(
                    "Hook didn't capture activations. Check `layer_idx`."
                )

            logits_teacher = out.logits.detach()
            hidden_proj = proj(hidden)
            logit_lens = do_logit_lens(model=base, hidden=hidden)
            logit_lens_proj = do_logit_lens(model=base, hidden=hidden_proj)

            with no_autocast(accelerator):
                loss = kl_loss_from_logits(
                    logit_lens_proj, logits_teacher, batch["attention_mask"]
                )

            accelerator.backward(loss)
            if config.clip_grad_norm is not None:
                accelerator.clip_grad_norm_(proj.parameters(), config.clip_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                mean_prob = calc_mean_prob(logit_lens)
                mean_prob_proj = calc_mean_prob(logit_lens_proj)
                mean_prob_true = calc_mean_prob(logits_teacher)

                assert (
                    mean_prob.dim() == 1
                    and len(mean_prob) == model.get_output_embeddings().weight.shape[0]
                ), (
                    mean_prob.dim(),
                    len(mean_prob),
                    model.get_output_embeddings().weight.shape[0],
                )
                accum_mean_prob.append(mean_prob.cpu())
                accum_mean_prob_proj.append(mean_prob_proj.cpu())
                accum_mean_prob_true.append(mean_prob_true.cpu())
                if accelerator.sync_gradients:
                    accum_mean_prob = (
                        torch.vstack(accum_mean_prob).mean(0).to(accelerator.device)
                    )
                    mean_prob = accelerator.reduce(accum_mean_prob, reduction="mean")
                    accum_mean_prob_proj = (
                        torch.vstack(accum_mean_prob_proj)
                        .mean(0)
                        .to(accelerator.device)
                    )
                    mean_prob_proj = accelerator.reduce(
                        accum_mean_prob_proj, reduction="mean"
                    )
                    accum_mean_prob_true = (
                        torch.vstack(accum_mean_prob_true)
                        .mean(0)
                        .to(accelerator.device)
                    )
                    mean_prob_true = accelerator.reduce(
                        accum_mean_prob_true, reduction="mean"
                    )

                    mean_dist = Categorical(probs=mean_prob)  # batch [e-s]
                    mean_proj_dist = Categorical(probs=mean_prob_proj)
                    mean_true_dist = Categorical(probs=mean_prob_true)

                    logit_lens_bias = torch.distributions.kl_divergence(
                        mean_true_dist, mean_dist
                    )
                    logit_lens_proj_bias = torch.distributions.kl_divergence(
                        mean_true_dist, mean_proj_dist
                    )

                    accelerator.log(
                        {
                            "train/logit_lens_bias": logit_lens_bias,
                            "train/logit_lens_proj_bias": logit_lens_proj_bias,
                        }
                    )

                    accum_mean_prob = []
                    accum_mean_prob_proj = []
                    accum_mean_prob_true = []

        incremented = False
        if step % config.grad_accum_steps == 0:
            incremented = True

            global_step += 1

        if incremented and step != 0 and global_step % config.log_every_steps == 0:
            lr = sched.get_last_lr()[0]
            step_tokens = (
                accelerator.gather_for_metrics(batch["attention_mask"].float().sum())
                .sum()
                .item()
            )
            accelerator.print(
                f"step {global_step:6d} | loss(kl) {loss.item():.4f} | lr {lr:.3e}"
            )
            accelerator.log(
                {
                    "train/loss_kl": loss.item(),
                    "train/lr": lr,
                    "train/step_tokens": step_tokens,
                    "train/global_step": global_step,
                    "train/epoch": global_step / updates_per_epoch,
                },
                step=global_step,
            )

    # --- Eval after the epoch ---
    eval_kl = evaluate_kl(model, proj, eval_loader, accelerator, config.layer_idx)
    accelerator.print(f"[eval@epoch1] mean token KL: {eval_kl:.4f}")
    accelerator.log(
        {"eval/mean_token_kl": eval_kl, "train/global_step": global_step},
        step=global_step,
    )

    # --- Save projection once at the end ---
    save_projection(accelerator, proj, config, config.layer_idx)

    # Cleanup
    handle.remove()
    accelerator.end_training()


if __name__ == "__main__":
    main()
