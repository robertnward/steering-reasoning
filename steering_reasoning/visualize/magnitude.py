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


def get_qwen2_5_math_7b_temp_1_model_groups(
    seed: int,
) -> Tuple[List[ModelGroup], List[Model]]:
    other_models = [
        Model(
            label=r"Base Model",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/base/temp_1.0_top_p_1.0",
            name=r"Base Model",
            ckpt=None,
        ),
        Model(
            label="Layer-15",
            pkl_path="/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-0_lr-0.01/checkpoint-314/vanilla/temp_1.0_top_p_1.0",
            name="Layer-15",
            ckpt="314",
        ),
    ]

    pkl_paths = [
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_-0.5/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_-1/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0.1/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0.5/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0.75/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0.9/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0.95/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_0.98/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_1/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_1.05/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_1.1/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_1.2/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_1.5/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_2/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_3/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_4/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_5/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_7/temp_1.0_top_p_1.0",
        f"/from_s3/results/Qwen2.5-Math-7B/deepscaler/steering-layer-15/seed-{seed}_lr-0.01/checkpoint-314/factor_10/temp_1.0_top_p_1.0",
    ]

    model_groups = []

    for pkl_path in pkl_paths:
        match = re.search(r"factor_(-?\d+\.\d+|-?\d+)", pkl_path)
        if match is None:
            continue
        else:
            factor = match.group(1)
        model_groups.append(
            ModelGroup(
                title=f"Qwen2.5-Math-7B. Factor-{factor}",
                models=[
                    Model(
                        label=f"factor_{factor}",
                        pkl_path=pkl_path.format(seed=seed),
                        name=f"factor_{factor}",
                        ckpt=None,
                    )
                ]
                + other_models,
            )
        )

    return model_groups, other_models


@pyrallis.wrap()
def run(config: Config):
    init_savedir = os.path.join(config.savedir, "magnitude")

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
            "Qwen2.5-Math-7B-temp-1",
        ],
        [
            get_qwen2_5_math_7b_temp_1_model_groups(seed=0),
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
                regex_pattern=r"factor_(-?\d+\.\d+|-?\d+)",
                xlabel="Magnitude Factor",
            )
            plot_benchmark_panels(
                data=data,
                savedir=os.path.join(savedir, name),
                other_models=other_models,
                regex_pattern=r"factor_(-?\d+\.\d+|-?\d+)",
            )


if __name__ == "__main__":
    run()
