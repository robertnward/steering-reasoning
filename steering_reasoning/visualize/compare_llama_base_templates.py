import os
from dataclasses import dataclass

import pandas as pd

from steering_reasoning.eval.eval import Problem, Solution  # noqa
from steering_reasoning.visualize.utils.common import dict_to_benchmark_df
from steering_reasoning.visualize.utils.defs import Model, ModelGroup
from steering_reasoning.visualize.utils.load_data import run_for_model_groups


@dataclass
class Config:
    savedir: str

    def __post_init__(self):
        os.makedirs(self.savedir, exist_ok=True)


def run():
    benches = [
        "AIME2025",
        "AIME_2024",
        "AMC-23",
        "MATH-500",
        "Minerva-Math",
        "OlympiadBench",
    ]
    for temp in [0.0, 1.0]:
        model_groups = [
            ModelGroup(
                title=f"Init template. Temp {temp}",
                models=[
                    Model(
                        name=f"Init template. Temp {temp}",
                        pkl_path=f"/from_s3/results/init_template/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"Init template. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"Qwen-Math Simplified template. Temp {temp}",
                models=[
                    Model(
                        name=f"Qwen-Math Simplified template. Temp {temp}",
                        pkl_path=f"/from_s3/results/qwen_math_template_simplified/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"Qwen-Math Simplified template. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_template/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template w spec tokens. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template w spec tokens. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_w_spec_tokens_template/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template w spec tokens. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template only final. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template only final. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_only_final_template/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template only final. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template only final 2. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template only final 2. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_only_final_template2/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template only final 2. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template first. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template first. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_template_first/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template first. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template only final first. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template only final first. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_only_final_template_first/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template only final first. Temp {temp}",
                    )
                ],
            ),
            ModelGroup(
                title=f"R1 template only final 2 first. Temp {temp}",
                models=[
                    Model(
                        name=f"R1 template only final 2 first. Temp {temp}",
                        pkl_path=f"/from_s3/results/r1_only_final_template2_first/temp_{temp}_top_p_1.0",
                        ckpt=None,
                        label=f"R1 template only final 2 first. Temp {temp}",
                    )
                ],
            ),
        ]

        config = Config(savedir="results")

        (
            accuracies_dict,
            avg_accuracies_dict,
            all_lens_dict,
            all_correct_lens_dict,
            all_incorrect_lens_dict,
            _,
        ) = run_for_model_groups(
            model_groups=model_groups,
            benches=benches,
            average_across_benches={
                "acc": ["MATH-500", "Minerva-Math", "OlympiadBench"],
                "avg_acc": ["AIME2025", "AIME_2024", "AMC-23"],
            },
        )

        savedir = os.path.join(config.savedir, "tables", f"temp_{temp}_top_p_1.0")
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
            results=all_incorrect_lens_dict,
            csv_path=os.path.join(savedir, "incorrect.csv"),
        )

    with pd.option_context(
        "display.max_colwidth",
        None,  # no ellipsis in long strings
        "display.width",
        None,  # or a big int like 2000
        "display.expand_frame_repr",
        False,  # keep each row on one line
    ):
        print("Temp 1.0")

        df = pd.read_csv(
            "results/tables/temp_1.0_top_p_1.0/avg_accuracies.csv", index_col=["Model"]
        ).drop("Setup", axis=1)
        df["Average"] = df.mean(1)
        print(df)

        print("Temp 0.0")

        df = pd.read_csv(
            "results/tables/temp_0.0_top_p_1.0/avg_accuracies.csv", index_col=["Model"]
        ).drop("Setup", axis=1)
        df["Average"] = df.mean(1)
        print(df)


if __name__ == "__main__":
    run()
