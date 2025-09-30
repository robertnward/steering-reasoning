from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Model:
    label: str
    pkl_path: Optional[str]
    name: str

    ckpt: Optional[str]


@dataclass
class ModelGroup:
    title: str
    models: List[Model]


class SetupMapping:
    def __getitem__(self, x):
        setup_mapping = {
            "steering": "Steering",
            "Base Model": "Base",
            "full_model": "Full-Tune",
        }
        if x in setup_mapping:
            return setup_mapping[x]
        else:
            return x
