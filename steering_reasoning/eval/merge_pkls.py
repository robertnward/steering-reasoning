#!/usr/bin/env python3
"""
Concatenate split .pkl shards found in the deepest subdirectory(ies) of `savedir`,
grouping by bench_name in filenames shaped like:
    "{bench_name}_{split_id}_{total_splits}.pkl"

It verifies ds_idx uniqueness and full coverage [0, len-1], then writes the merged
lists to the sibling directory whose last path segment is `savedir.name.split("_")[0]`,
preserving the same inner subdirectory structure.

Usage:
    python merge_bench_pkls.py --savedir /path/to/results_8/sub1/sub2
Requires:
    pip install pyrallis
"""

import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyrallis

from steering_reasoning.eval.eval import Problem, Solution  # noqa

FNAME_RE = re.compile(r"^(?P<bench>.+)_(?P<split>\d+)_(?P<total>\d+)\.pkl$")


@dataclass
class Config:
    """CLI config parsed by pyrallis."""

    savedir: Path  # Root directory that may contain nested subdirs with shards


def parse_name(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    return (
        m.group("bench"),
        int(m.group("split")),
        int(m.group("total")),
    )


def find_deepest_dirs(root: Path) -> List[Path]:
    """
    Find the deepest directory(ies) under `root`. If multiple are tied for max depth,
    return them all. If `root` itself is a leaf, return [root].
    """
    deepest: List[Path] = []
    max_depth = -1
    for dirpath, dirnames, _ in os.walk(root):
        p = Path(dirpath)
        depth = len(p.relative_to(root).parts) if p != root else 0
        if not dirnames:  # leaf
            if depth > max_depth:
                deepest = [p]
                max_depth = depth
            elif depth == max_depth:
                deepest.append(p)
    if not deepest:
        deepest = [root]
    return deepest


def load_pickle_list(path: Path) -> List[Any]:
    # NOTE: Unpickling arbitrary files can execute code; only use trusted data.
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"{path} did not contain a list (got {type(obj).__name__})")
    return obj


def extract_ds_idx(item: Any) -> int:
    """
    Tries common patterns to read ds_idx:
      - attribute: item.ds_idx
      - mapping key: item['ds_idx'] or item.get('ds_idx')
    """
    if hasattr(item, "ds_idx"):
        return int(getattr(item, "ds_idx"))
    if isinstance(item, dict):
        if "ds_idx" in item:
            return int(item["ds_idx"])
        val = item.get("ds_idx")
        if val is not None:
            return int(val)
    raise AttributeError("Item lacks ds_idx attribute/key")


def verify_indices(items: List[Any], where: str) -> None:
    """Ensure ds_idx are unique and exactly 0..len(items)-1."""
    idxs = []
    for i, it in enumerate(items):
        try:
            idxs.append(extract_ds_idx(it))
        except Exception as e:
            raise AssertionError(f"{where}: element #{i} has no readable ds_idx: {e}")
    n = len(items)
    idx_set = set(idxs)
    expected = set(range(n))
    if idx_set != expected:
        missing = sorted(expected - idx_set)
        extra = sorted(idx_set - expected)
        raise AssertionError(
            f"{where}: ds_idx check failed.\n"
            f"- expected 0..{n - 1} ({n} items)\n"
            f"- missing {len(missing)} (e.g., {missing[:10]})\n"
            f"- extra {len(extra)} (e.g., {extra[:10]})"
        )


def plan_output_dir(savedir: Path, leaf: Path) -> Path:
    """
    Compute the output directory, which is a sibling of `savedir` where the last
    path segment is replaced with savedir.name.split('_')[0], and we preserve the
    relative path from savedir -> leaf.
    """
    base_name = savedir.name.split("_")[0]
    out_root = savedir.with_name(base_name)
    rel = leaf.relative_to(savedir) if leaf != savedir else Path(".")
    return out_root / rel


def process_leaf(savedir: Path, leaf: Path) -> List[Tuple[str, Path]]:
    """
    Process a single deepest directory: group shards by bench, concatenate, verify,
    and write {bench}.pkl files. Returns list of (bench_name, output_file) written.
    """
    groups: Dict[str, List[Tuple[Tuple[str, int, int], Path]]] = defaultdict(list)

    for p in sorted(leaf.glob("*.pkl")):
        parsed = parse_name(p.name)
        if parsed:
            groups[parsed[0]].append((parsed, p))

    if not groups:
        raise FileNotFoundError(f"No matching '*_split_total.pkl' files in {leaf}")

    written: List[Tuple[str, Path]] = []
    out_dir = plan_output_dir(savedir, leaf)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench, entries in sorted(groups.items()):
        # Sort by split id for reproducibility
        entries_sorted = sorted(entries, key=lambda t: t[0][1])
        # Optional sanity: ensure all total_splits agree
        totals = {e[0][2] for e in entries_sorted}
        if len(totals) != 1:
            raise AssertionError(
                f"{leaf}: '{bench}' has mixed total_splits values: {sorted(totals)}"
            )
        total = next(iter(totals))
        # Helpful notes if incomplete or weird splits:
        seen_splits = {e[0][1] for e in entries_sorted}
        if seen_splits != set(range(min(seen_splits), max(seen_splits) + 1)) or len(
            seen_splits
        ) != len(entries_sorted):
            print(
                f"Warning: '{bench}' in {leaf} has non-contiguous or duplicate split IDs: {sorted(seen_splits)}"
            )
        if total != len(entries_sorted):
            print(
                f"Note: '{bench}' expected {total} splits but found {len(entries_sorted)} in {leaf}."
            )

        concatenated: List[Any] = []
        for (_bench, _split_id, _total), path in entries_sorted:
            shard = load_pickle_list(path)
            concatenated.extend(shard)

        verify_indices(concatenated, where=f"{leaf.name}/{bench}")

        out_path = out_dir / f"{bench}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(concatenated, f, protocol=pickle.HIGHEST_PROTOCOL)
        written.append((bench, out_path))
        print(f"Wrote {bench}: {len(concatenated)} items -> {out_path}")

    return written


@pyrallis.wrap()
def main(cfg: Config) -> None:
    savedir = cfg.savedir.resolve()
    if not savedir.exists() or not savedir.is_dir():
        raise NotADirectoryError(f"{savedir} is not a directory.")

    leaves = find_deepest_dirs(savedir)
    print(f"Found {len(leaves)} deepest director{'y' if len(leaves) == 1 else 'ies'}:")
    for d in leaves:
        print(" -", d)

    all_written: List[Tuple[str, Path]] = []
    for leaf in leaves:
        written = process_leaf(savedir, leaf)
        all_written.extend(written)

    if not all_written:
        print("No outputs were written.")
    else:
        print("\nSummary:")
        for bench, outp in all_written:
            print(f" - {bench} -> {outp}")


if __name__ == "__main__":
    main()  # pyrallis parses CLI and calls main(cfg)
