import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pyrallis
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from loguru import logger

SPLIT_DIR_RE = re.compile(r"^split_(\d+)_(\d+)$")


@dataclass
class Config:
    # Required
    savedir: str

    # Options
    sort_by: str = "prompt_idx"
    overwrite: bool = False

    # Advanced
    split_dir_regex: str = field(
        default="^split_(\\d+)_(\\d+)$",
        metadata={"help": "Regex for split directories"},
    )
    # If True, also include the path of each merged dataset in the summary logs.
    verbose_summary: bool = False


def find_split_groups(root: str, split_re: re.Pattern) -> Dict[str, List[str]]:
    """
    Walk `root` and return a mapping: parent_dir -> [split_dir_paths...],
    where each `split_dir_path` is a directory named like 'split_i_N'.
    """
    groups: Dict[str, List[str]] = {}
    for dirpath, dirnames, _files in os.walk(root):
        for name in dirnames:
            if split_re.match(name):
                parent = os.path.abspath(dirpath)
                groups.setdefault(parent, []).append(os.path.join(parent, name))
    # Sort splits per parent for reproducibility
    for parent in groups:
        groups[parent].sort()
    return groups


def _as_dataset(obj) -> Dataset:
    """Normalize a Dataset or DatasetDict to a single Dataset."""
    if isinstance(obj, Dataset):
        return obj
    if isinstance(obj, DatasetDict):
        if "train" in obj:
            return obj["train"]
        return concatenate_datasets(list(obj.values()))
    raise TypeError(f"Unsupported dataset type: {type(obj)}")


def load_any_dataset(path: str) -> Dataset:
    """
    Try loading a dataset from a local directory.

    Priority:
      1) load_from_disk(path) — for datasets saved via `save_to_disk`
      2) load_dataset(path, split="train") — for local dataset scripts
      3) load_dataset(path) — and then normalize to a single split
    """
    # 1) Arrow dataset on disk
    try:
        ds = load_from_disk(path)
        return _as_dataset(ds)
    except Exception as e:
        logger.debug(f"load_from_disk failed for '{path}': {e}")

    # 2) Local dataset loader script or folder with default 'train' split
    try:
        ds = load_dataset(path, split="train")
        return _as_dataset(ds)
    except Exception as e:
        logger.debug(f"load_dataset(path, split='train') failed for '{path}': {e}")

    # 3) Try without split, then normalize
    try:
        ds = load_dataset(path)
        return _as_dataset(ds)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from '{path}': {e}") from e


def compute_output_dir(savedir: str, parent_of_splits: str) -> str:
    """
    Mirror the directory structure *up to the parent that contains the split_i_N* folders,
    rooted at '{savedir}_combined'.

    Example:
        savedir = /data/root
        parent_of_splits = /data/root/A/B           (which contains split_0_N, split_1_N, ...)
        ->
        output_dir = /data/root_combined/A/B
    """
    root = os.path.abspath(savedir)
    combined_root = f"{root}_combined"
    rel = os.path.relpath(os.path.abspath(parent_of_splits), root)
    if rel == os.curdir:
        # Splits are directly under savedir
        return combined_root
    return os.path.join(combined_root, rel)


def merge_and_sort(splits: List[str], sort_by: str) -> Dataset:
    """Load all split datasets, concatenate, and sort."""
    datasets_list: List[Dataset] = []
    for p in splits:
        logger.info(f"Loading dataset from: {p}")
        ds = load_any_dataset(p)
        datasets_list.append(ds)

    if not datasets_list:
        raise RuntimeError("No datasets loaded from the provided split directories.")

    logger.info(f"Concatenating {len(datasets_list)} datasets")
    merged = concatenate_datasets(datasets_list)

    if sort_by not in merged.column_names:
        raise ValueError(
            f"Column '{sort_by}' not found in merged dataset. Available: {merged.column_names}"
        )

    logger.info(f"Sorting by '{sort_by}'")
    merged = merged.sort(sort_by)
    return merged


@pyrallis.wrap()
def main(config: Config):
    logger.info("Starting merge of split datasets")
    savedir = os.path.abspath(config.savedir)
    if not os.path.isdir(savedir):
        logger.error(f"'{savedir}' is not a directory.")
        raise SystemExit(1)

    split_re = re.compile(config.split_dir_regex)
    groups = find_split_groups(savedir, split_re)

    if not groups:
        logger.error(
            f"No directories matching pattern '{config.split_dir_regex}' were found under: {savedir}"
        )
        raise SystemExit(2)

    logger.info(f"Found {len(groups)} group(s) of split directories")
    if config.verbose_summary:
        for parent, splits in groups.items():
            logger.info(f"Group parent: {parent}")
            for s in splits:
                logger.info(f"  - {s}")

    merged_outputs: List[Tuple[str, int]] = []

    for parent, splits in groups.items():
        try:
            merged = merge_and_sort(splits, config.sort_by)
        except Exception as e:
            logger.exception(f"Failed to merge group at '{parent}': {e}")
            continue

        outdir = compute_output_dir(savedir, parent)
        logger.info(f"Computed output directory: {outdir}")

        if os.path.exists(outdir):
            if config.overwrite:
                logger.warning(
                    f"Output directory exists; removing because --overwrite is set: {outdir}"
                )
                shutil.rmtree(outdir)
            else:
                logger.error(
                    f"Output directory already exists (use --overwrite to replace): {outdir}"
                )
                continue

        os.makedirs(os.path.dirname(outdir), exist_ok=True)
        logger.info(f"Saving merged dataset ({len(merged)} rows) to: {outdir}")
        merged.save_to_disk(outdir)
        merged_outputs.append((outdir, len(merged)))

    if not merged_outputs:
        logger.error("No groups were successfully merged and saved.")
        raise SystemExit(3)

    logger.info("Merge completed successfully.")
    for outdir, nrows in merged_outputs:
        logger.info(f"Saved merged dataset with {nrows} rows -> {outdir}")


if __name__ == "__main__":
    main()
