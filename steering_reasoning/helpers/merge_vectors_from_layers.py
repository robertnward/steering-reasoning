import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pyrallis


@dataclass
class Config:
    model_path: str
    save_path: str


def merge_steering_vectors(paths: List[str], savepath: str):
    all_steering_vectors = []
    for path in paths:
        steering_vectors = np.load(path)
        layer_idx = int(path.split("/")[5].rsplit("-", 1)[-1])
        assert len(all_steering_vectors) == layer_idx, (
            len(all_steering_vectors),
            layer_idx,
        )
        all_steering_vectors.append(steering_vectors[layer_idx])

    all_steering_vectors = np.vstack(all_steering_vectors)
    np.save(os.path.join(savepath, "steering_vectors.npy"), all_steering_vectors)


def get_qwen2_5_math_7b_model_paths(seed: int):
    return [
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-0/seed-{seed}_lr-0.003/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-1/seed-{seed}_lr-0.007/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-2/seed-{seed}_lr-0.005/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-3/seed-{seed}_lr-0.007/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-4/seed-{seed}_lr-0.007/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-5/seed-{seed}_lr-0.007/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-6/seed-{seed}_lr-0.007/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-7/seed-{seed}_lr-0.007/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-8/seed-{seed}_lr-0.01/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-9/seed-{seed}_lr-0.01/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-10/seed-{seed}_lr-0.01/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-11/seed-{seed}_lr-0.01/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-12/seed-{seed}_lr-0.01/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-13/seed-{seed}_lr-0.01/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-14/seed-{seed}_lr-0.01/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-16/seed-{seed}_lr-0.01/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-17/seed-{seed}_lr-0.01/checkpoint-159/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-18/seed-{seed}_lr-0.03/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-19/seed-{seed}_lr-0.05/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-20/seed-{seed}_lr-0.05/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-21/seed-{seed}_lr-0.1/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-22/seed-{seed}_lr-0.1/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-23/seed-{seed}_lr-0.01/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-24/seed-{seed}_lr-0.01/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-25/seed-{seed}_lr-0.1/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-{seed}_lr-0.1/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-{seed}_lr-0.1/checkpoint-265/steering_vectors.npy",
    ]


def get_llama3_1_8b_chat_model_paths(seed: int):
    return [
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-0/seed-{seed}_lr-0.0005/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-1/seed-{seed}_lr-0.0005/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-2/seed-{seed}_lr-0.0005/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-3/seed-{seed}_lr-0.0005/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-4/seed-{seed}_lr-0.0007/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-5/seed-{seed}_lr-0.0007/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-6/seed-{seed}_lr-0.0007/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-7/seed-{seed}_lr-0.0007/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-8/seed-{seed}_lr-0.001/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-9/seed-{seed}_lr-0.001/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-10/seed-{seed}_lr-0.001/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-11/seed-{seed}_lr-0.002/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-12/seed-{seed}_lr-0.001/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-13/seed-{seed}_lr-0.002/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-14/seed-{seed}_lr-0.001/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-15/seed-{seed}_lr-0.002/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-16/seed-{seed}_lr-0.002/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-17/seed-{seed}_lr-0.002/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-18/seed-{seed}_lr-0.002/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-19/seed-{seed}_lr-0.002/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-20/seed-{seed}_lr-0.002/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-21/seed-{seed}_lr-0.002/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-22/seed-{seed}_lr-0.002/checkpoint-212/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-23/seed-{seed}_lr-0.002/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-24/seed-{seed}_lr-0.002/checkpoint-106/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-25/seed-{seed}_lr-0.005/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-26/seed-{seed}_lr-0.005/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-27/seed-{seed}_lr-0.005/checkpoint-265/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-28/seed-{seed}_lr-0.005/checkpoint-106/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-29/seed-{seed}_lr-0.007/checkpoint-314/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-{seed}_lr-0.007/checkpoint-106/steering_vectors.npy",
        f"/from_s3/steering_vectors/llama3.1-8b-chat/deepscaler/steering-layer-31/seed-{seed}_lr-0.005/checkpoint-159/steering_vectors.npy",
    ]


@pyrallis.wrap()
def run(config: Config):
    # Qwen2.5-Math-7B paths
    for seed in [0, 1]:
        qwen2_5_math_7b_savepath = os.path.join(
            config.save_path,
            "Qwen2.5-Math-7B",
            "deepscaler",
            "steering-layer-merged",
            f"seed-{seed}",
        )

        os.makedirs(qwen2_5_math_7b_savepath, exist_ok=True)

        qwen2_5_math_7b_paths = get_qwen2_5_math_7b_model_paths(seed)

        merge_steering_vectors(
            paths=qwen2_5_math_7b_paths, savepath=qwen2_5_math_7b_savepath
        )

    llama3_1_8b_chat_savepath = os.path.join(
        config.save_path,
        "llama3.1-8b-chat",
        "deepscaler",
        "steering-layer-merged",
        "seed-0",
    )
    os.makedirs(llama3_1_8b_chat_savepath, exist_ok=True)

    llama3_1_8b_chat_paths = get_llama3_1_8b_chat_model_paths(0)

    merge_steering_vectors(
        paths=llama3_1_8b_chat_paths, savepath=llama3_1_8b_chat_savepath
    )


if __name__ == "__main__":
    run()
