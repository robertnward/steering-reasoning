import io
import logging
import pathlib
import sys
import time
import zipfile
from dataclasses import asdict
from typing import Any, List

import torch
import wandb
from loguru import logger


def pad_sequences(
    sequences: List[torch.Tensor], pad_length: int, pad_value: int
) -> torch.Tensor:
    padded_sequences = []
    for seq in sequences:
        assert seq.dim() == 1, seq.shape

        padding_needed = pad_length - len(seq)
        assert padding_needed >= 0, padding_needed
        padded_seq = torch.hstack(
            [
                seq,
                torch.full(
                    size=(padding_needed,),
                    fill_value=pad_value,
                    dtype=seq.dtype,
                    device=seq.device,
                ),
            ]
        )
        assert padded_seq.dim() == 1, padded_seq.shape
        padded_sequences.append(padded_seq)

    return torch.vstack(padded_sequences)


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def wandb_init(config, run_id: str | None = None):
    if run_id is None:
        run = wandb.init(
            project=config.project,
            group=config.group,
            job_type=config.job_type,
            entity=config.entity,
            name=config.name,
            config=asdict(config),
            save_code=False,
        )
    else:
        run = wandb.init(
            project=config.project,
            entity=config.entity,
            id=run_id,
            resume="must",
        )

    return run


def is_sublist(main_list: List[Any], sub_list: List[Any], return_idx: bool):
    """
    Check if sub_list is a contiguous sublist of main_list.

    Args:
        main_list (list): The list to be searched.
        sub_list (list): The list to search for.

    Returns:
        bool: True if sub_list is found as a contiguous block in main_list, False otherwise.
    """
    if not sub_list:
        # An empty list is considered a sublist of any list.
        return True

    len_main = len(main_list)
    len_sub = len(sub_list)

    # Iterate over main_list with a window of size len_sub
    for i in range(len_main - len_sub + 1):
        if main_list[i : i + len_sub] == sub_list:
            if not return_idx:
                return True
            else:
                return True, i
    if not return_idx:
        return False
    else:
        return False, -1


def set_logger(verbose: bool) -> None:
    """
    Set the logging level of loguru.
    The effect of this function is global, and it should
    be called only once in the main function
    """
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    logger.disable("math_verify")
    logging.getLogger("math_verify").setLevel(logging.CRITICAL)


class Timeit:
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_gpu = torch.cuda.Event(enable_timing=True)
            self.end_gpu = torch.cuda.Event(enable_timing=True)
            self.start_gpu.record()
        self.start_cpu = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            self.end_gpu.record()
            torch.cuda.synchronize()
            self.elapsed_time_gpu = self.start_gpu.elapsed_time(self.end_gpu) / 1000
        else:
            self.elapsed_time_gpu = -1.0
        self.elapsed_time_cpu = time.time() - self.start_cpu


def dir_to_zip_bytes(dir_path: str | pathlib.Path) -> bytes:
    """Return a ZIP-compressed Bytes object that contains everything under *dir_path*."""
    dir_path = pathlib.Path(dir_path).expanduser().resolve()

    buffer = io.BytesIO()
    # ZIP_DEFLATED gives you classic .zip compression
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in dir_path.rglob("*"):  # walk recursively
            if entry.is_file():  # skip dirs & symlinks
                zf.write(entry, arcname=entry.relative_to(dir_path))

    buffer.seek(0)  # rewind
    return buffer.getvalue()


def _safe_extract_zip(zf: zipfile.ZipFile, dest: pathlib.Path):
    for member in zf.infolist():
        target_path = dest / member.filename
        if not target_path.resolve().is_relative_to(dest):
            raise RuntimeError(f"Blocked suspicious path: {member.filename}")
        zf.extract(member, path=dest)


def zip_bytes_to_dir(data: bytes, dest: str | pathlib.Path) -> None:
    """Inflate *data* (a ZIP archive held in bytes) into *dest* directory."""
    dest = pathlib.Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        _safe_extract_zip(zf, dest)
