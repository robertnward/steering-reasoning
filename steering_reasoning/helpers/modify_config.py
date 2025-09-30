import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyrallis
from loguru import logger


@dataclass
class Config:
    model_dir: str
    cfg_path: str
    new_arch: Optional[str] = None

    proj_type: Optional[str] = None
    head_idx: Optional[int] = None
    add_place: Optional[str] = None
    auto_map_causal_lm: Optional[str] = None
    path_patch_layer_index: Optional[str] = None
    patch_path: Optional[str] = None
    add_layer: Optional[bool] = None

    def __post_init__(self):
        assert self.proj_type is None or self.proj_type in [
            "head",
            "group",
            "group3_and",
            "group4_and",
            "group6_and",
            "inverse_head",
            "inverse_group",
            "inverse_group7_and",
            "inverse_group7_and1_and",
            "inverse_group7_and6_and",
            "inverse_group7_and1_and2_and",
            "inverse_group7_and1_and5_and",
            "q_proj",
            "k_proj",
            "v_proj",
            "all_v_proj",
            "inverse_q_proj",
            "inverse_k_proj",
            "inverse_v_proj",
            "inverse_all_v_proj",
            "all_attn",
            "all_layer",
            "skip_attn",
            "skip_layer",
            "no_patch",
            "base",
        ], self.proj_type

        assert self.patch_path is None or self.patch_path in [
            "residual",
            "mlp",
            "all",
        ], self.patch_path


@pyrallis.wrap()
def run(config: Config):
    model_dir = Path(config.model_dir)
    cfg_path = model_dir / config.cfg_path

    cfg = json.loads(cfg_path.read_text())
    if config.new_arch is not None:
        cfg["architectures"] = [config.new_arch]
    if config.proj_type is not None:
        cfg["proj_type"] = config.proj_type
    if config.head_idx is not None:
        cfg["head_idx"] = config.head_idx
    if config.add_place is not None:
        cfg["add_place"] = config.add_place
    if config.path_patch_layer_index is not None:
        cfg["path_patch_layer_index"] = config.path_patch_layer_index
    if config.patch_path is not None:
        cfg["patch_path"] = config.patch_path
    if config.add_layer is not None:
        cfg["num_hidden_layers"] += 1
    if config.auto_map_causal_lm is not None:
        cfg["auto_map"] = {"AutoModelForCausalLM": config.auto_map_causal_lm}

    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    logger.info(
        f"architectures = {cfg['architectures']}; proj_type = {config.proj_type}; head_idx = {config.head_idx}; add_place = {config.add_place}; patch_path = {config.patch_path}; add_layer = {config.add_layer}"
    )


if __name__ == "__main__":
    run()
