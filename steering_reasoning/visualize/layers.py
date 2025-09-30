import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import pyrallis

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.common import (
    dict_to_benchmark_df,
    filter_benchmarks_for_models,
    plot_benchmark_curves,
    plot_benchmark_panels,
    save_best_ckpts,
    select_best_ckpt_per_name,
)
from steering_reasoning.visualize.utils.defs import Model, ModelGroup
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    seed: int

    loaddir: str
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def get_qwen2_5_math_7b_temp_0_model_groups(
    seed: int,
) -> Tuple[List[ModelGroup], List[Model]]:
    other_models = [
        Model(
            label="Base Model",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_0.0_top_p_1.0",
            name="Base Model",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. $\tau=1$",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_1.0_top_p_1.0",
            name=r"Base Model. $\tau=1$",
            ckpt=None,
        ),
        Model(
            label="Steering",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-212/append_to_False/temp_0.0_top_p_1.0",
            name="Steering",
            ckpt=None,
        ),
    ]
    pkl_paths = [
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-0/seed-{seed}_lr-0.003/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-1/seed-{seed}_lr-0.007/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-2/seed-{seed}_lr-0.005/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-3/seed-{seed}_lr-0.007/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-4/seed-{seed}_lr-0.007/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-5/seed-{seed}_lr-0.007/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-6/seed-{seed}_lr-0.007/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-7/seed-{seed}_lr-0.007/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-8/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-9/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-10/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-11/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-12/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-13/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-14/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-16/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-17/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-18/seed-{seed}_lr-0.03/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-19/seed-{seed}_lr-0.05/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-20/seed-{seed}_lr-0.05/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-21/seed-{seed}_lr-0.1/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-22/seed-{seed}_lr-0.1/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-23/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-24/seed-{seed}_lr-0.01/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-25/seed-{seed}_lr-0.1/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-{seed}_lr-0.1/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-{seed}_lr-0.1/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"steering-layer-(\d+)", pkl_path)
        if match is None:
            continue
        else:
            layer = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"Qwen2.5-Math-7B. Layer-{layer}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt=ckpt, seed=seed),
                        name=f"steering-layer-{layer}",
                        ckpt=f"ckpt-{ckpt}",
                    )
                    for ckpt in [106, 159, 212, 265, 314]
                ]
                + other_models,
            )
        )

    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-temp-0",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-{ckpt}/append_to_False/temp_0.0_top_p_1.0",
                    name="steering-temp-0",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )
    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-27 append_to",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-0_lr-0.1_append_to_True/checkpoint-{ckpt}/append_to_True/temp_0.0_top_p_1.0",
                    name="steering-layer-27 on to",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )
    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-26 append_to",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-0_lr-0.1_append_to_True/checkpoint-{ckpt}/append_to_True/temp_0.0_top_p_1.0",
                    name="steering-layer-26 on to",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )

    return model_groups, other_models


def get_qwen2_5_math_7b_temp_1_model_groups(
    seed: int,
) -> Tuple[List[ModelGroup], List[Model]]:
    other_models = [
        Model(
            label="Base Model",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_0.0_top_p_1.0",
            name="Base Model",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. $\tau=1$",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_1.0_top_p_1.0",
            name=r"Base Model. $\tau=1$",
            ckpt=None,
        ),
        Model(
            label="Steering",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-265/temp_1.0_top_p_1.0",
            name="Steering",
            ckpt=None,
        ),
        Model(
            label="Base Model. Append To",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/append_to_True/temp_0.0_top_p_1.0",
            name="Base Model. Append To",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. Append To. $\tau=1$",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/append_to_True/temp_1.0_top_p_1.0",
            name=r"Base Model. Append To. $\tau=1$",
            ckpt=None,
        ),
        # Model(
        #     label="steering-layer-26 on to",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-0_lr-0.1_append_to_True/checkpoint-314/append_to_True/temp_1.0_top_p_1.0",
        #     name="steering-layer-26 on to",
        #     ckpt=None,
        # ),
        # Model(
        #     label="steering-layer-27 on to",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-0_lr-0.1_append_to_True/checkpoint-314/append_to_True/temp_1.0_top_p_1.0",
        #     name="steering-layer-27 on to",
        #     ckpt=None,
        # ),
        # Model(
        #     label="steering-0.01-14-16",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.01_in_14-16/checkpoint-159/temp_1.0_top_p_1.0",
        #     name="steering-0.01-14-16",
        #     ckpt=None,
        # ),
        # Model(
        #     label="steering-0.005-14-16",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.007_in_14-16/checkpoint-314/temp_1.0_top_p_1.0",
        #     name="steering-0.005-14-16",
        #     ckpt=None,
        # ),
        # Model(
        #     label="steering-0.007-14-16",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.005_in_14-16/checkpoint-212/temp_1.0_top_p_1.0",
        #     name="steering-0.007-14-16",
        #     ckpt=None,
        # ),
        # Model(
        #     label="steering-0.001-23-24",
        #     pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.001_ext_23-24/checkpoint-314/temp_1.0_top_p_1.0",
        #     name="steering-0.001-23-24",
        #     ckpt=None,
        # ),
    ]
    pkl_paths = [
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-0/seed-{seed}_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-1/seed-{seed}_lr-0.007/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-2/seed-{seed}_lr-0.005/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-3/seed-{seed}_lr-0.007/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-4/seed-{seed}_lr-0.007/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-5/seed-{seed}_lr-0.007/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-6/seed-{seed}_lr-0.007/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-7/seed-{seed}_lr-0.007/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-8/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-9/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-10/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-11/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-12/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-13/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-14/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-16/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-17/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-18/seed-{seed}_lr-0.03/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-19/seed-{seed}_lr-0.05/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-20/seed-{seed}_lr-0.05/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-21/seed-{seed}_lr-0.1/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-22/seed-{seed}_lr-0.1/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-23/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-24/seed-{seed}_lr-0.01/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-25/seed-{seed}_lr-0.1/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-{seed}_lr-0.1/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-{seed}_lr-0.1/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"steering-layer-(\d+)", pkl_path)
        if match is None:
            continue
        else:
            layer = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"Qwen2.5-Math-7B. Layer-{layer}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt=ckpt, seed=seed),
                        name=f"steering-layer-{layer}",
                        ckpt=f"ckpt-{ckpt}",
                    )
                    for ckpt in [106, 159, 212, 265, 314]
                ]
                + other_models,
            )
        )
    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-27 append_to",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-0_lr-0.1_append_to_True/checkpoint-{ckpt}/append_to_True/temp_1.0_top_p_1.0",
                    name="steering-layer-27 on to",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )
    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-26 append_to",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-0_lr-0.1_append_to_True/checkpoint-{ckpt}/append_to_True/temp_1.0_top_p_1.0",
                    name="steering-layer-26 on to",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )

    # model_groups.append(
    #     ModelGroup(
    #         title="Qwen2.5-Math-7B. steering-0.01-14-16",
    #         models=[
    #             Model(
    #                 label=f"ckpt-{ckpt}",
    #                 pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.01_in_14-16/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
    #                 name="!steering-0.01-14-16",
    #                 ckpt=ckpt,
    #             )
    #             for ckpt in [106, 159, 212, 265, 314]
    #         ]
    #         + other_models[:2],
    #     )
    # )
    # model_groups.append(
    #     ModelGroup(
    #         title="Qwen2.5-Math-7B. steering-0.007-14-16",
    #         models=[
    #             Model(
    #                 label=f"ckpt-{ckpt}",
    #                 pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.007_in_14-16/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
    #                 name="!steering-0.007-14-16",
    #                 ckpt=ckpt,
    #             )
    #             for ckpt in [106, 159, 212, 265, 314]
    #         ]
    #         + other_models[:2],
    #     )
    # )
    # model_groups.append(
    #     ModelGroup(
    #         title="Qwen2.5-Math-7B. steering-0.005-14-16",
    #         models=[
    #             Model(
    #                 label=f"ckpt-{ckpt}",
    #                 pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.005_in_14-16/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
    #                 name="!steering-0.005-14-16",
    #                 ckpt=ckpt,
    #             )
    #             for ckpt in [106, 159, 212, 265, 314]
    #         ]
    #         + other_models[:2],
    #     )
    # )
    # model_groups.append(
    #     ModelGroup(
    #         title="Qwen2.5-Math-7B. steering-0.001-ext-23-24",
    #         models=[
    #             Model(
    #                 label=f"ckpt-{ckpt}",
    #                 pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0_lr-0.001_ext_23-24/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
    #                 name="!steering-0.001-ext-23-24",
    #                 ckpt=ckpt,
    #             )
    #             for ckpt in [106, 159, 212, 265, 314]
    #         ]
    #         + other_models[:2],
    #     )
    # )

    return model_groups, other_models


def get_qwen2_5_math_7b_temp_1_lora_model_groups() -> Tuple[
    List[ModelGroup], List[Model]
]:
    other_models = [
        Model(
            label="Base Model",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_0.0_top_p_1.0",
            name="Base Model",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. $\tau=1$",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_1.0_top_p_1.0",
            name=r"Base Model. $\tau=1$",
            ckpt=None,
        ),
        Model(
            label="Steering",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-265/temp_1.0_top_p_1.0",
            name="Steering",
            ckpt=None,
        ),
        Model(
            label="Base Model. Append To",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/append_to_True/temp_0.0_top_p_1.0",
            name="Base Model. Append To",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. Append To. $\tau=1$",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/append_to_True/temp_1.0_top_p_1.0",
            name=r"Base Model. Append To. $\tau=1$",
            ckpt=None,
        ),
    ]
    pkl_paths = [
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-0/seed-0_lr-0.005/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-1/seed-0_lr-0.005/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-2/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-3/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-4/seed-0_lr-0.005/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-5/seed-0_lr-0.005/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-6/seed-0_lr-0.005/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-7/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-8/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-9/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-10/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-11/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-12/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-13/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-14/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-15/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-16/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-17/seed-0_lr-0.003/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-18/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-19/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-20/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-21/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-22/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-23/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-24/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-25/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-26/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1-layer-27/seed-0_lr-0.001/checkpoint-{ckpt}/temp_1.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"lora-1-layer-(\d+)", pkl_path)
        if match is None:
            continue
        else:
            layer = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"Qwen2.5-Math-7B. Layer-{layer}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt=ckpt),
                        name=f"steering-layer-{layer}",
                        ckpt=f"ckpt-{ckpt}",
                    )
                    for ckpt in [106, 159, 212, 265, 314]
                    if os.path.exists(pkl_path.format(ckpt=ckpt))
                ]
                + other_models,
            )
        )
    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-27 append_to",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-27/seed-0_lr-0.1_append_to_True/checkpoint-{ckpt}/append_to_True/temp_1.0_top_p_1.0",
                    name="steering-layer-27 on to",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )
    model_groups.append(
        ModelGroup(
            title="Qwen2.5-Math-7B. steering-26 append_to",
            models=[
                Model(
                    label=f"ckpt-{ckpt}",
                    pkl_path=f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-26/seed-0_lr-0.1_append_to_True/checkpoint-{ckpt}/append_to_True/temp_1.0_top_p_1.0",
                    name="steering-layer-26 on to",
                    ckpt=f"ckpt-{ckpt}",
                )
                for ckpt in [106, 159, 212, 265, 314]
            ]
            + other_models[:2],
        )
    )

    return model_groups, other_models


def get_llama3_1_8b_chat_temp_0_model_groups() -> Tuple[List[ModelGroup], List[Model]]:
    other_models = [
        Model(
            label="Base Model",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
            name="Base Model",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. $\tau=1$",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
            name=r"Base Model. $\tau=1$",
            ckpt=None,
        ),
        Model(
            label="Steering",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering/seed-0/checkpoint-265/temp_1.0_top_p_1.0",
            name="Steering",
            ckpt=None,
        ),
        Model(
            label="Base Model. Append Step",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/append_to_True/temp_0.0_top_p_1.0",
            name="Base Model. Append Step",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. Append Step. $\tau=1$",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/append_to_True/temp_1.0_top_p_1.0",
            name=r"Base Model. Append Step. $\tau=1$",
            ckpt=None,
        ),
    ]

    pkl_paths = [
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-0/seed-0_lr-0.0005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-1/seed-0_lr-0.0005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-2/seed-0_lr-0.0005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-3/seed-0_lr-0.0005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-4/seed-0_lr-0.0007/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-5/seed-0_lr-0.0007/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-6/seed-0_lr-0.0007/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-7/seed-0_lr-0.0007/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-8/seed-0_lr-0.001/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-9/seed-0_lr-0.001/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-10/seed-0_lr-0.001/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-11/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-12/seed-0_lr-0.001/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-13/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-14/seed-0_lr-0.001/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-15/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-16/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-17/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-18/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-19/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-20/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-21/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-22/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-23/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-24/seed-0_lr-0.002/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-25/seed-0_lr-0.005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-26/seed-0_lr-0.005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-27/seed-0_lr-0.005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-28/seed-0_lr-0.005/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-29/seed-0_lr-0.007/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-{}/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-31/seed-0_lr-0.005/checkpoint-{}/temp_0.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"steering-layer-(\d+)", pkl_path)
        if match is None:
            continue
        else:
            layer = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"llama3.1-8b-chat. Layer-{layer}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt),
                        name=f"steering-layer-{layer}",
                        ckpt=f"ckpt-{ckpt}",
                    )
                    for ckpt in [106, 159, 212, 265, 314]
                ]
                + other_models,
            )
        )

    return model_groups, other_models


def get_llama3_1_8b_chat_temp_1_model_groups() -> Tuple[List[ModelGroup], List[Model]]:
    other_models = [
        Model(
            label="Base Model",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
            name="Base Model",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. $\tau=1$",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
            name=r"Base Model. $\tau=1$",
            ckpt=None,
        ),
        Model(
            label="Steering",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering/seed-0/checkpoint-265/temp_1.0_top_p_1.0",
            name="Steering",
            ckpt=None,
        ),
        Model(
            label="Base Model. Append Step",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/append_to_True/temp_0.0_top_p_1.0",
            name="Base Model. Append Step",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. Append Step. $\tau=1$",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/append_to_True/temp_1.0_top_p_1.0",
            name=r"Base Model. Append Step. $\tau=1$",
            ckpt=None,
        ),
    ]

    pkl_paths = [
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-0/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-1/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-2/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-3/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-4/seed-0_lr-0.0007/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-5/seed-0_lr-0.0007/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-6/seed-0_lr-0.0007/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-7/seed-0_lr-0.0007/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-8/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-9/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-10/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-11/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-12/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-13/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-14/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-15/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-16/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-17/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-18/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-19/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-20/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-21/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-22/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-23/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-24/seed-0_lr-0.002/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-25/seed-0_lr-0.005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-26/seed-0_lr-0.005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-27/seed-0_lr-0.005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-28/seed-0_lr-0.005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-29/seed-0_lr-0.007/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-30/seed-0_lr-0.007/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering-layer-31/seed-0_lr-0.005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"steering-layer-(\d+)", pkl_path)
        if match is None:
            continue
        else:
            layer = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"llama3.1-8b-chat. Layer-{layer}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt),
                        name=f"steering-layer-{layer}",
                        ckpt=f"ckpt-{ckpt}",
                    )
                    for ckpt in [106, 159, 212, 265, 314]
                ]
                + other_models,
            )
        )

    return model_groups, other_models


def get_llama3_1_8b_chat_temp_1_lora_model_groups() -> Tuple[
    List[ModelGroup], List[Model]
]:
    other_models = [
        Model(
            label="Base Model",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
            name="Base Model",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. $\tau=1$",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/temp_1.0_top_p_1.0",
            name=r"Base Model. $\tau=1$",
            ckpt=None,
        ),
        Model(
            label="Steering",
            pkl_path="/from_s3/results/llama3.1-8b-chat/deepscaler/steering/seed-0/checkpoint-265/temp_1.0_top_p_1.0",
            name="Steering",
            ckpt=None,
        ),
        Model(
            label="Base Model. Append Step",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/append_to_True/temp_0.0_top_p_1.0",
            name="Base Model. Append Step",
            ckpt=None,
        ),
        Model(
            label=r"Base Model. Append Step. $\tau=1$",
            pkl_path="/from_s3/results/llama3.1-8b-chat/base/append_to_True/temp_1.0_top_p_1.0",
            name=r"Base Model. Append Step. $\tau=1$",
            ckpt=None,
        ),
    ]

    pkl_paths = [
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-0/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-1/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-2/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-3/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-4/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-5/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-6/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-7/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-8/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-9/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-10/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-11/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-12/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-13/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-14/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-15/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-16/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-17/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-18/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-19/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-20/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-21/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-22/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-23/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-24/seed-0_lr-0.0005/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-25/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-26/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-27/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-28/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-29/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-30/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1-layer-31/seed-0_lr-0.0005/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"lora-1-layer-(\d+)", pkl_path)
        if match is None:
            continue
        else:
            layer = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"llama3.1-8b-chat. Layer-{layer}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt),
                        name=f"steering-layer-{layer}",
                        ckpt=f"ckpt-{ckpt}",
                    )
                    for ckpt in [106, 159, 212, 265, 314]
                    if os.path.exists(pkl_path.format(ckpt))
                ]
                + other_models,
            )
        )

    return model_groups, other_models


@pyrallis.wrap()
def run(config: Config):
    init_savedir = os.path.join(config.savedir, "layers")

    benches = [
        "AIME2025",
        "AIME_2024",
        "AMC-23",
        "MATH-500",
        "Minerva-Math",
        "OlympiadBench",
    ]

    for model_groups_name, (model_groups, other_models) in zip(
        [
            "Qwen2.5-Math-7B-temp-0",
            "Qwen2.5-Math-7B-temp-1",
            "Qwen2.5-Math-7B-temp-1-lora",
            "llama3.1-8b-chat-temp-0",
            "llama3.1-8b-chat-temp-1",
            "llama3.1-8b-chat-temp-1-lora",
        ],
        [
            get_qwen2_5_math_7b_temp_0_model_groups(seed=0),
            get_qwen2_5_math_7b_temp_1_model_groups(seed=0),
            get_qwen2_5_math_7b_temp_1_lora_model_groups(),
            get_llama3_1_8b_chat_temp_0_model_groups(),
            get_llama3_1_8b_chat_temp_1_model_groups(),
            get_llama3_1_8b_chat_temp_1_lora_model_groups(),
        ],
    ):
        config.savedir = os.path.join(init_savedir, model_groups_name)
        os.makedirs(config.savedir, exist_ok=True)
        # Final plots and tables
        (
            accuracies_dict,
            avg_accuracies_dict,
            all_lens_dict,
            all_correct_lens_dict,
            all_max_correct_lens_dict,
            all_incorrect_lens_dict,
        ) = run_for_model_groups(
            model_groups=model_groups,
            benches=benches,
            average_across_benches={
                "acc": ["MATH-500", "Minerva-Math", "OlympiadBench"],
                "avg_acc": ["AIME2025", "AIME_2024", "AMC-23"],
            },
        )

        savedir = os.path.join(config.savedir, "tables", "temp_1.0_top_p_1.0")
        os.makedirs(savedir, exist_ok=True)
        dict_to_benchmark_df(
            results=accuracies_dict, csv_path=os.path.join(savedir, "accuracies.csv")
        )
        dict_to_benchmark_df(
            results=avg_accuracies_dict,
            csv_path=os.path.join(savedir, "avg_accuracies.csv"),
        )
        dict_to_benchmark_df(
            results=all_lens_dict, csv_path=os.path.join(savedir, "lens.csv")
        )
        dict_to_benchmark_df(
            results=all_correct_lens_dict, csv_path=os.path.join(savedir, "correct.csv")
        )
        dict_to_benchmark_df(
            results=all_max_correct_lens_dict,
            csv_path=os.path.join(savedir, "max_correct.csv"),
        )
        dict_to_benchmark_df(
            results=all_incorrect_lens_dict,
            csv_path=os.path.join(savedir, "incorrect.csv"),
        )

        final_accuracies_dict = {
            "AIME2025": avg_accuracies_dict["AIME2025"],
            "AIME_2024": avg_accuracies_dict["AIME_2024"],
            "AMC-23": avg_accuracies_dict["AMC-23"],
            "MATH-500": accuracies_dict["MATH-500"],
            "Minerva-Math": accuracies_dict["Minerva-Math"],
            "OlympiadBench": accuracies_dict["OlympiadBench"],
            "Average": accuracies_dict["Average"],
        }

        savedir = os.path.join(config.savedir, "best-ckpts")
        os.makedirs(savedir, exist_ok=True)

        best_models = list(
            select_best_ckpt_per_name(final_accuracies_dict["Average"]).keys()
        )
        save_best_ckpts(best_models=best_models, savedir=savedir)

        filtered_accuracies = filter_benchmarks_for_models(
            final_accuracies_dict, selected_models=best_models
        )
        filtered_all_lens = filter_benchmarks_for_models(
            all_lens_dict, selected_models=best_models
        )
        filtered_all_correct_lens = filter_benchmarks_for_models(
            all_correct_lens_dict, selected_models=best_models
        )
        filtered_all_max_correct_lens = filter_benchmarks_for_models(
            all_max_correct_lens_dict, selected_models=best_models
        )
        filtered_all_incorrect_lens = filter_benchmarks_for_models(
            all_incorrect_lens_dict, selected_models=best_models
        )
        for data, name in zip(
            [
                filtered_accuracies,
                filtered_all_lens,
                filtered_all_correct_lens,
                filtered_all_max_correct_lens,
                filtered_all_incorrect_lens,
            ],
            [
                "accuracies",
                "all_lens",
                "all_correct_lens",
                "all_max_correct_lens",
                "all_incorrect_lens",
            ],
        ):
            plot_benchmark_curves(
                data=data,
                savedir=os.path.join(savedir, name),
                other_models=other_models,
                regex_pattern=r"^steering-layer-(\d+)",
                xlabel="Layer",
            )
            plot_benchmark_panels(
                data=data,
                savedir=os.path.join(savedir, name),
                other_models=other_models,
                regex_pattern=r"^steering-layer-(\d+)",
            )


if __name__ == "__main__":
    run()
