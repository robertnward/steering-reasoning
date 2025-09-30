import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pyrallis

from steering_reasoning.eval.eval import Problem, Solution  # noqa  # noqa  # noqa
from steering_reasoning.visualize.utils.common import (
    dict_to_benchmark_df,
    filter_benchmarks_for_models,
    name_to_label,
    save_best_ckpts,
    select_best_ckpt_per_name,
)
from steering_reasoning.visualize.utils.defs import Model, ModelGroup, SetupMapping
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    seed: int

    loaddir: str
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def add_reference_accuracies(
    accuracies_dict: Dict[str, Dict[Model, Dict[str, float]]],
) -> Dict[str, Dict[Model, Dict[str, float]]]:
    # Oat-Zero https://arxiv.org/pdf/2503.20783
    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-Math-1.5B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 20.0,
        "std": 0.0,
    }
    accuracies_dict["AMC-23 PASS@1"][
        Model(name="Qwen2.5-Math-1.5B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 53.0,
        "std": 0.0,
    }
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-Math-1.5B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 74.2,
        "std": 0.0,
    }
    accuracies_dict["Minerva-Math"][
        Model(name="Qwen2.5-Math-1.5B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 25.7,
        "std": 0.0,
    }
    accuracies_dict["OlympiadBench"][
        Model(name="Qwen2.5-Math-1.5B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 37.6,
        "std": 0.0,
    }
    accuracies_dict["Average"][
        Model(name="Qwen2.5-Math-1.5B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": np.nan,
        "std": np.nan,
    }

    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-Math-7B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 43.3,
        "std": 0.0,
    }
    accuracies_dict["AMC-23 PASS@1"][
        Model(name="Qwen2.5-Math-7B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 62.7,
        "std": 0.0,
    }
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-Math-7B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 80.0,
        "std": 0.0,
    }
    accuracies_dict["Minerva-Math"][
        Model(name="Qwen2.5-Math-7B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 30.1,
        "std": 0.0,
    }
    accuracies_dict["OlympiadBench"][
        Model(name="Qwen2.5-Math-7B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 41.0,
        "std": 0.0,
    }
    accuracies_dict["Average"][
        Model(name="Qwen2.5-Math-7B", label="Oat-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": np.nan,
        "std": np.nan,
    }

    accuracies_dict["AIME2025 PASS@1"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": np.nan, "std": np.nan}
    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": 6.7, "std": 0.0}
    accuracies_dict["AIME_2024 AVG@32"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": 4.2, "std": 0.0}
    accuracies_dict["AMC-23 PASS@1"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": 35.0, "std": 0.0}
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": 59.0, "std": 0}
    accuracies_dict["Minerva-Math"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": 20.2, "std": 0.0}
    accuracies_dict["OlympiadBench"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": 21.0, "std": 0.0}
    accuracies_dict["Average"][
        Model(name="Qwen2.5-1.5B", label="SimpleRL-Zoo", ckpt=None, pkl_path=None)
    ] = {"mean": np.nan, "std": np.nan}

    accuracies_dict["AIME2025 PASS@1"][
        Model(name="Qwen2.5-1.5B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": 1.0, "std": 0.0}
    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-1.5B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": 3.5, "std": 0.0}
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-1.5B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": 58.0, "std": 0}
    accuracies_dict["Average"][
        Model(name="Qwen2.5-1.5B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": np.nan, "std": np.nan}

    accuracies_dict["AIME_2024 PASS@1"][
        Model(
            name="Qwen2.5-7B",
            label="Open-Reasoner (from Oat-Zero)",
            ckpt=None,
            pkl_path=None,
        )
    ] = {"mean": 13.3, "std": 0.0}
    accuracies_dict["AMC-23 PASS@1"][
        Model(
            name="Qwen2.5-7B",
            label="Open-Reasoner (from Oat-Zero)",
            ckpt=None,
            pkl_path=None,
        )
    ] = {"mean": 54.2, "std": 0.0}
    accuracies_dict["MATH-500"][
        Model(
            name="Qwen2.5-7B",
            label="Open-Reasoner (from Oat-Zero)",
            ckpt=None,
            pkl_path=None,
        )
    ] = {"mean": 82.2, "std": 0.0}
    accuracies_dict["Minerva-Math"][
        Model(
            name="Qwen2.5-7B",
            label="Open-Reasoner (from Oat-Zero)",
            ckpt=None,
            pkl_path=None,
        )
    ] = {"mean": 31.6, "std": 0.0}
    accuracies_dict["OlympiadBench"][
        Model(
            name="Qwen2.5-7B",
            label="Open-Reasoner (from Oat-Zero)",
            ckpt=None,
            pkl_path=None,
        )
    ] = {"mean": 47.9, "std": 0.0}
    accuracies_dict["Average"][
        Model(
            name="Qwen2.5-7B",
            label="Open-Reasoner (from Oat-Zero)",
            ckpt=None,
            pkl_path=None,
        )
    ] = {"mean": np.nan, "std": np.nan}

    accuracies_dict["AIME2025 PASS@1"][
        Model(name="Qwen2.5-7B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": 15.6, "std": 0.0}
    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-7B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": 17.9, "std": 0.0}
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-7B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": 81.4, "std": 0.0}
    accuracies_dict["Average"][
        Model(name="Qwen2.5-7B", label="Open-Reasoner", ckpt=None, pkl_path=None)
    ] = {"mean": np.nan, "std": np.nan}

    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 20.0,
        "std": 0.0,
    }
    accuracies_dict["AIME_2024 AVG@32"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 15.6,
        "std": 0.0,
    }
    accuracies_dict["AMC-23 PASS@1"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 62.5,
        "std": 0.0,
    }
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 78.2,
        "std": 0.0,
    }
    accuracies_dict["Minerva-Math"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 38.6,
        "std": 0.0,
    }
    accuracies_dict["OlympiadBench"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 40.4,
        "std": 0.0,
    }
    accuracies_dict["Average"][
        Model(name="Qwen2.5-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": np.nan,
        "std": np.nan,
    }

    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 40.0,
        "std": 0.0,
    }
    accuracies_dict["AIME_2024 AVG@32"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 24.0,
        "std": 0.0,
    }
    accuracies_dict["AMC-23 PASS@1"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 70.0,
        "std": 0.0,
    }
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 80.2,
        "std": 0.0,
    }
    accuracies_dict["Minerva-Math"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 37.5,
        "std": 0.0,
    }
    accuracies_dict["OlympiadBench"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": 39.0,
        "std": 0.0,
    }
    accuracies_dict["Average"][
        Model(name="Qwen2.5-Math-7B", label="SimpleRL-Zero", ckpt=None, pkl_path=None)
    ] = {
        "mean": np.nan,
        "std": np.nan,
    }

    accuracies_dict["AIME_2024 PASS@1"][
        Model(name="Qwen2.5-7B", label="R1-Distill", ckpt=None, pkl_path=None)
    ] = {
        "mean": 33.0,
        "std": 0.0,
    }
    accuracies_dict["AMC-23 PASS@1"][
        Model(name="Qwen2.5-7B", label="R1-Distill", ckpt=None, pkl_path=None)
    ] = {
        "mean": 68.4,
        "std": 0.0,
    }
    accuracies_dict["MATH-500"][
        Model(name="Qwen2.5-7B", label="R1-Distill", ckpt=None, pkl_path=None)
    ] = {
        "mean": 88.1,
        "std": 0.0,
    }
    accuracies_dict["Minerva-Math"][
        Model(name="Qwen2.5-7B", label="R1-Distill", ckpt=None, pkl_path=None)
    ] = {
        "mean": 35.9,
        "std": 0.0,
    }
    accuracies_dict["OlympiadBench"][
        Model(name="Qwen2.5-7B", label="R1-Distill", ckpt=None, pkl_path=None)
    ] = {
        "mean": 47.7,
        "std": 0.0,
    }
    accuracies_dict["Average"][
        Model(name="Qwen2.5-7B", label="R1-Distill", ckpt=None, pkl_path=None)
    ] = {
        "mean": np.nan,
        "std": np.nan,
    }

    return accuracies_dict


@pyrallis.wrap()
def run(config: Config):
    config.savedir = os.path.join(config.savedir, "models_setups")
    os.makedirs(config.savedir, exist_ok=True)

    benches = [
        "AIME2025",
        "AIME_2024",
        "AMC-23",
        "MATH-500",
        "Minerva-Math",
        "OlympiadBench",
    ]

    setup_mapping = SetupMapping()

    pkl_paths = [
        "/from_s3/results/Qwen2.5-1.5B-Instruct/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-1.5B-Instruct/deepscaler/steering/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-1.5B-Instruct/deepscaler/full_model/seed-0_lr-1e-06/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-1.5B/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-1.5B/deepscaler/steering/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-1.5B/deepscaler/full_model/seed-0_lr-2e-06/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-7B-Instruct/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-7B-Instruct/deepscaler/steering/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-7B-Instruct/deepscaler/full_model/seed-0_lr-1e-06/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/lora-1/seed-0_lr-0.0001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-Math-7B/deepscaler/full_model/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-1.5B/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-1.5B/deepscaler/steering/seed-0_lr-0.001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-1.5B/deepscaler/full_model/seed-0_lr-1e-06/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-7B/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-7B/deepscaler/steering/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-7B/deepscaler/full_model/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-14B/base/temp_0.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-14B/deepscaler/steering/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/Qwen2.5-14B/deepscaler/full_model/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/base/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/steering/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/lora-1/seed-0_lr-0.0001/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b-chat/deepscaler/full_model/seed-0/checkpoint-{}/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b/base/append_to_False/temp_0.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b/deepscaler/steering/seed-0_lr-0.0003_last/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        # "/from_s3/results/llama3.1-8b/deepscaler/full_model/seed-0_lr-2e-06_last/checkpoint-{}/append_to_False/temp_1.0_top_p_1.0",
        "/from_s3/results/llama3.1-8b/deepscaler/full_model/seed-0_lr-1e-06_last/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_23_mlp/seed-0_lr-0.01/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_24_input_layernorm/seed-0_lr-0.01/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_24_self_attn/seed-0_lr-0.01/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_24_post_attention_layernorm/seed-0_lr-0.01/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_24_mlp/seed-0_lr-0.01/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_25_input_layernorm/seed-0_lr-0.05/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_25_self_attn/seed-0_lr-0.05/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_25_post_attention_layernorm/seed-0_lr-0.05/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_25_mlp/seed-0_lr-0.05/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_27_self_attn/seed-0_lr-0.1/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_27_post_attention_layernorm/seed-0_lr-0.05/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/Qwen2.5-Math-7B/deepscaler/add_place_27_post_attention_layernorm_single_place/seed-0_lr-0.05/checkpoint-{}/temp_1.0_top_p_1.0",
        # "/from_s3/results/llama3.1-8b-chat/deepscaler/add_place_31_self_attn/seed-0_lr-0.005/checkpoint-{}/temp_1.0_top_p_1.0",
    ]
    model_groups = []

    for pkl_path in pkl_paths:
        model_name = Path(pkl_path).parts[3]
        if model_name == "llama3.1-8b-chat":
            model_name = "LLaMa3.1-8B-It"
        elif model_name == "llama3.1-8b":
            model_name = "LLaMa3.1-8B"
        if "checkpoint" in pkl_path:
            setup = Path(pkl_path).parts[5]
        else:
            setup = "Base Model"
        model_groups.append(
            ModelGroup(
                title=f"{model_name}. {setup}",
                models=[
                    Model(
                        label=f"ckpt-{ckpt}",
                        pkl_path=pkl_path.format(ckpt),
                        name=f"{model_name}|{setup}",
                        ckpt=f"ckpt-{ckpt}" if ckpt is not None else None,
                    )
                    for ckpt in (
                        [106, 159, 212, 265, 314] if setup != "base" else [None]  # type: ignore
                    )
                ],
            )
        )

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

    final_accuracies_dict = {
        "AIME2025 AVG@32": avg_accuracies_dict["AIME2025"],
        "AIME_2024 AVG@32": avg_accuracies_dict["AIME_2024"],
        "AMC-23 AVG@32": avg_accuracies_dict["AMC-23"],
        "MATH-500": accuracies_dict["MATH-500"],
        "Minerva-Math": accuracies_dict["Minerva-Math"],
        "OlympiadBench": accuracies_dict["OlympiadBench"],
        "Average": accuracies_dict["Average"],
    }

    best_models = list(
        select_best_ckpt_per_name(final_accuracies_dict["Average"]).keys()
    )
    save_best_ckpts(best_models=best_models, savedir=config.savedir)

    final_accuracies_dict = {
        "AIME2025 PASS@1": accuracies_dict["AIME2025"],
        "AIME_2024 PASS@1": accuracies_dict["AIME_2024"],
        "AMC-23 PASS@1": accuracies_dict["AMC-23"],
        **final_accuracies_dict,
    }
    filtered_accuracies = name_to_label(
        filter_benchmarks_for_models(
            final_accuracies_dict, selected_models=best_models
        ),
        setup_mapping=setup_mapping,
    )
    filtered_accuracies = add_reference_accuracies(accuracies_dict=filtered_accuracies)
    filtered_all_lens = name_to_label(
        filter_benchmarks_for_models(all_lens_dict, selected_models=best_models),
        setup_mapping=setup_mapping,
    )
    filtered_all_correct_lens = name_to_label(
        filter_benchmarks_for_models(
            all_correct_lens_dict, selected_models=best_models
        ),
        setup_mapping=setup_mapping,
    )
    filtered_all_incorrect_lens = name_to_label(
        filter_benchmarks_for_models(
            all_incorrect_lens_dict, selected_models=best_models
        ),
        setup_mapping=setup_mapping,
    )
    savedir = os.path.join(config.savedir, "tables")
    os.makedirs(savedir, exist_ok=True)
    dict_to_benchmark_df(
        results=filtered_accuracies, csv_path=os.path.join(savedir, "accuracies.csv")
    )
    dict_to_benchmark_df(
        results=filtered_all_lens, csv_path=os.path.join(savedir, "lens.csv")
    )
    dict_to_benchmark_df(
        results=filtered_all_correct_lens, csv_path=os.path.join(savedir, "correct.csv")
    )
    dict_to_benchmark_df(
        results=filtered_all_incorrect_lens,
        csv_path=os.path.join(savedir, "incorrect.csv"),
    )


if __name__ == "__main__":
    run()
