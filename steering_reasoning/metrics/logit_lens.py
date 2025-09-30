import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrallis
import torch
from accelerate.utils import set_seed
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_reasoning.metrics.plots import (
    calc_stable_rank,
    draw_norms,
    draw_pairwise_cossims,
    draw_singular_values,
)


@dataclass
class Config:
    model_path: str
    steering_vectors_dir: str
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)
        os.makedirs(os.path.join(self.savedir, "dot_sims"), exist_ok=True)
        os.makedirs(os.path.join(self.savedir, "cos_sims"), exist_ok=True)


def get_prompt(tokens: List[str]):
    tokens = ["'" + tok + "'" for tok in tokens]
    tokens = ", ".join(tokens)

    return (
        """
**Task**
You will receive a list of single-word or short-phrase **tokens** that may be written in different languages.

1. **Detect** the language of every token.
2. **Translate** each **non-English** token into English **before** you attempt any clustering.
3. Show all translations in a “Translations” section (see template).
4. Using the translated meanings (not the original spelling) for guidance, **group** the tokens into logical clusters that share a clear, unifying **concept** (e.g., “medical terms,” “colors,” “personal names”). Language is irrelevant—focus only on meaning.
5. Give each cluster a concise **topic label** (in English).
6. If a token does not fit any cluster, place it under **“Unclassified.”**
7. Create clusters only when at least **two** tokens belong to the same topic; otherwise keep singletons in “Unclassified.”
8. Present your answer in **Markdown** using **exactly** the template below—first the translations table, then the clusters.

**Output template**

```markdown
## Translations
| Original | English | Language |
|----------|---------|----------|
| token1   | apple   | French   |
| token2   | …       | Arabic   |
| *(leave English tokens out of the table)* |

## Topic 1: <topic-label>
- token A  
- token B  
- …

## Topic 2: <topic-label>
- token C  
- token D  
- …

## Unclassified
- token E  
- token F  
```

**Tokens to analyze**

```
"""
        + tokens
        + """
```
    """
    )


def send_batched_requests(
    messages: List[List[dict]],  # same structure as before
    batch_size: int,
    token: Optional[str] = os.environ.get("OPENAI_TOKEN", None),
    url: Optional[str] = os.environ.get("OPENAI_ADDRESS", None),
    model: str = "openai/gpt-4.1",
    temperature: float = 0.0,
    n: int = 1,
) -> List[str]:
    """
    Synchronous version of `send_batched_requests`.

    Parameters
    ----------
    messages : List[List[dict]]
        A list where each element is the 'messages' array you would feed
        to the Chat Completions API.
    batch_size : int
        How many separate 'messages' groups to process in one batch.
    Other params are unchanged.
    """
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    def send_single_request(client: httpx.Client, msg: List[dict]) -> List[str]:
        payload = {
            "model": model,
            "messages": msg,
            "temperature": temperature,
            "max_tokens": 16_000,
            "n": n,
            "stream": False,
        }

        print(f"Sending request: {payload}")

        while True:
            try:
                resp = client.post(url, headers=headers, json=payload)
                response_json = resp.json()
                print(response_json)
                resp.raise_for_status()
                return [
                    choice["message"]["content"] for choice in response_json["choices"]
                ]
            except (httpx.HTTPError, httpx.TimeoutException, KeyError) as exc:
                print(f"Retrying after error: {exc}")
                time.sleep(3)

    results: List[str] = []
    with httpx.Client(timeout=60.0) as client:
        num_batches = (len(messages) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch = messages[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            for msg in batch:
                results.extend(send_single_request(client, msg))

    return results


def write_summarization(top_tokens, f):
    messages = [[{"role": "user", "content": get_prompt(tokens=top_tokens)}]]

    summarization = send_batched_requests(messages, batch_size=8)
    assert len(summarization) == 1, len(summarization)
    f.write("\n")
    f.write("-" * 50)
    f.write("\n")
    f.write(summarization[0])


def save_hist_grid(
    data: Sequence[Iterable[float]],
    savepath: str | Path,
    *,
    bins: int | Sequence[float] | str = "auto",
    rows: int = 4,
    figsize: Tuple[float, float] | None = None,
    dpi: int = 300,
    **hist_kw,
) -> Path:
    """
    Draw one histogram per inner list in *data*, arranged in a grid that
    always has `rows` rows, and save the figure to *savepath*.

    Parameters
    ----------
    data     : sequence of 1-D iterables
        The numeric datasets to plot.
    savepath : str | pathlib.Path
        Where to write the image (suffix decides format, e.g. .png, .pdf).
    bins     : int, sequence, str, optional
        Binning rule forwarded to ``plt.hist`` (default is ``'auto'``).
    rows     : int, optional
        Fixed number of subplot rows (default 4).
    figsize  : (w, h) tuple in inches, optional
        Overall figure size.  Defaults to (4 × cols, 3 × rows).
    dpi      : int, optional
        Resolution for raster formats (default 300).
    **hist_kw: other keyword arguments for ``plt.hist``
        E.g. ``color``, ``alpha``, ``density``.

    Returns
    -------
    pathlib.Path
        The resolved path of the file that was written.
    """
    data = list(data)  # allow any iterable
    n = len(data)
    if n == 0:
        raise ValueError("data must contain at least one inner sequence")

    cols = math.ceil(n / rows)
    if figsize is None:
        figsize = (4 * cols, 3 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # Plot each histogram
    for idx, (ax, arr) in enumerate(zip(axes_flat, data)):
        ax.hist(arr, bins=bins, **hist_kw)
        ax.set_title(f"Layer {idx}")
        max_abs = max(abs(np.min(arr)), abs(np.max(arr)))
        ax.set_yscale("log")
        ax.set_xlim(-1.1 * max_abs, 1.1 * max_abs)
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Count")

    # Hide unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    # Ensure parent directory exists and save
    savepath = Path(savepath).expanduser().resolve()
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # free memory

    return savepath


def top_tokens_table(
    top_tokens_idx: np.ndarray,
    top_tokens: List[str],
    cos_sims: np.ndarray,
    dot_sims: np.ndarray,
    savepath: str,
):
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    if cos_sims.ndim != 1 or dot_sims.ndim != 1 or top_tokens_idx.ndim != 1:
        raise ValueError("cos_sims and dot_sims must be 1-D arrays after squeezing.")
    if len(top_tokens) != len(top_tokens_idx):
        raise ValueError(
            f"Length of top_tokens ({len(top_tokens)}) must match size of top_tokens_idx ({len(top_tokens_idx)})."
        )
    if (
        top_tokens_idx.max(initial=-1) >= cos_sims.size
        or top_tokens_idx.max(initial=-1) >= dot_sims.size
        or top_tokens_idx.min(initial=0) < 0
    ):
        raise IndexError(
            "top_tokens_idx contains indices out of bounds for the similarity arrays."
        )

    selected_cos = cos_sims[top_tokens_idx].round(2)
    selected_dot = dot_sims[top_tokens_idx].round(2)
    # Build DataFrame
    df = pd.DataFrame(
        [selected_cos, selected_dot],
        index=["Cos. Sim.", "Dot Prod."],
        columns=[token.replace("\n", "\\n") for token in top_tokens],
    )

    df.to_csv(savepath, index=True)


@pyrallis.wrap()
def run(config: Config):
    set_seed(seed=config.seed, device_specific=False, deterministic=False)
    steering_vectors = np.load(
        os.path.join(config.steering_vectors_dir, "steering_vectors.npy")
    )
    steering_vectors = torch.from_numpy(steering_vectors).to(torch.bfloat16).cuda()
    print(steering_vectors.shape)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)

    os.makedirs(os.path.join(config.savedir, "stats"), exist_ok=True)
    draw_norms(
        steering_vectors=steering_vectors.float().cpu().numpy(),
        savedir=os.path.join(config.savedir, "stats"),
    )
    draw_pairwise_cossims(
        steering_vectors=steering_vectors.float().cpu().numpy(),
        savepath=os.path.join(config.savedir, "stats", "cossims.png"),
    )
    draw_singular_values(
        steering_vectors=steering_vectors.float().cpu().numpy(),
        savepath=os.path.join(config.savedir, "stats", "singular_values.png"),
    )
    stable_rank = calc_stable_rank(steering_vectors.float().cpu().numpy())
    logger.info(f"Stable rank: {stable_rank}")

    with torch.no_grad():
        unembed_matrix = model.get_output_embeddings().weight.data.cuda()

        all_dot_sims = []
        all_cos_sims = []
        for layer_idx, steering_vector in enumerate(steering_vectors):
            dot_sims = unembed_matrix @ steering_vector
            top_dot_sims, top_tokens_dot_sims_idx = torch.topk(dot_sims, 50)
            all_dot_sims.append(dot_sims.float().cpu().numpy())

            top_tokens_dot_sims = [tokenizer.decode(t) for t in top_tokens_dot_sims_idx]
            with open(
                os.path.join(config.savedir, "dot_sims", f"layer_{layer_idx}.txt"), "+w"
            ) as f:
                for sim, token in zip(top_dot_sims, top_tokens_dot_sims):
                    f.write(f"{sim.item():.4f} {token}\n")

                # write_summarization(top_tokens=top_tokens_dot_sims, f=f)

            cos_sims = (
                unembed_matrix / torch.linalg.norm(unembed_matrix, dim=-1, keepdim=True)
            ) @ (steering_vector / torch.linalg.norm(steering_vector))
            top_cos_sims, top_tokens_cos_sims_idx = torch.topk(cos_sims, 50)
            all_cos_sims.append(cos_sims.float().cpu().numpy())
            top_tokens_cos_sims = [tokenizer.decode(t) for t in top_tokens_cos_sims_idx]
            with open(
                os.path.join(config.savedir, "cos_sims", f"layer_{layer_idx}.txt"), "+w"
            ) as f:
                for sim, token in zip(top_cos_sims, top_tokens_cos_sims):
                    f.write(f"{sim.item():.4f} {token}\n")

                # write_summarization(top_tokens=top_tokens_cos_sims, f=f)

            top_tokens_table(
                top_tokens_idx=top_tokens_cos_sims_idx.cpu().numpy()[:10],
                top_tokens=top_tokens_cos_sims[:10],
                cos_sims=cos_sims.float().cpu().numpy(),
                dot_sims=dot_sims.float().cpu().numpy(),
                savepath=os.path.join(
                    config.savedir, "top_tokens_tables", f"layer_{layer_idx}_table.csv"
                ),
            )

    save_hist_grid(
        data=all_dot_sims,
        savepath=os.path.join(config.savedir, "dot_sims.png"),
        bins=100,
    )
    save_hist_grid(
        data=all_cos_sims,
        savepath=os.path.join(config.savedir, "cos_sims.png"),
        bins=100,
    )


if __name__ == "__main__":
    run()
