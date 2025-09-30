import os
from dataclasses import dataclass
from typing import List, Optional

import pyrallis
import torch


@dataclass
class Config:
    seed: int

    model_path: str
    batch_size: int
    max_seq_length: int

    dataset_path: str
    train_datasets: List[str]
    test_datasets: List[str]

    repetition_penalty: float
    stop_token: str
    top_k: int
    top_p: float
    generation_temperature: float

    savedir: str

    num_generations_for_train: int
    num_generations_for_test: int

    calc_type: str

    alpha_min: float
    alpha_max: float
    alpha_num_steps: int

    num_train_samples: Optional[int]
    num_test_samples: Optional[int]

    train_gen_batch_size: int
    test_gen_batch_size: int

    device: Optional[str]

    debug_on_single_sample: bool
    plot_sample_specific_results: bool

    check_consistency: bool
    num_reiterations_for_check_consistency: Optional[int]
    check_consistency_sample_size: Optional[int]

    hidden_dim: Optional[int]

    def __post_init__(self):
        assert self.calc_type in ["diff", "svm"], self.calc_type

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        assert self.num_generations_for_train > 1, self.num_generations_for_train

        subsubdirname = (
            f"train_gens_{self.num_generations_for_train}"
            + f"_test_gens_{self.num_generations_for_test}"
            + f"_train_samples_{self.num_train_samples}"
            + f"_test_samples_{self.num_test_samples}"
        )
        if self.debug_on_single_sample:
            subsubdirname = "single_sample_" + subsubdirname

        if self.check_consistency:
            subsubdirname = (
                f"check_consistency_{self.check_consistency_sample_size}_"
                + subsubdirname
            )

        model_name = os.path.basename(self.model_path)
        self.savedir = os.path.join(
            self.savedir,
            model_name,
            f"calc_type_{self.calc_type}",
            f"temp_{self.generation_temperature}",
            subsubdirname,
        )

        if self.check_consistency:
            assert self.num_reiterations_for_check_consistency is not None
            assert self.check_consistency_sample_size is not None

        os.makedirs(self.savedir, exist_ok=True)


@pyrallis.wrap()
def get_savedir(config: Config):
    print(config.savedir)


if __name__ == "__main__":
    get_savedir()
