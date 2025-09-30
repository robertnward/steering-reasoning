import os

import numpy as np
import torch
import torch.distributed

world_size = int(os.getenv("WORLD_SIZE", "1"))


def get_global_mean(values: torch.Tensor) -> float:
    # Calculate the mean reward for the current process
    local_sum = values.sum().item()

    if world_size == 1:
        return values.mean().item()

    num_rewards = torch.tensor(len(values), device=values.device)

    # Create a tensor to hold the global mean reward
    global_sum = torch.tensor(local_sum, device=values.device)

    # Collect mean rewards from all processes
    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_rewards, op=torch.distributed.ReduceOp.SUM)
    global_mean = (global_sum / num_rewards).item()

    return global_mean


def get_global_std(values: torch.Tensor, mean: float) -> float:
    local_sum = ((values - mean) ** 2).sum().item()

    if world_size == 1:
        return local_sum

    # Calculate the mean reward for the current process
    num_rewards = torch.tensor(len(values), device=values.device)

    # Create a tensor to hold the global mean reward
    global_sum = torch.tensor(local_sum, device=values.device)

    # Collect mean rewards from all processes
    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_rewards, op=torch.distributed.ReduceOp.SUM)
    global_mean = np.sqrt((global_sum / (num_rewards - 1)).item())

    return global_mean


def get_mean_std(
    tensor: torch.Tensor,
    use_global: bool,
) -> tuple[torch.tensor, torch.tensor]:
    if (
        use_global
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        # On each rank, compute local sums on GPU
        local_sum = torch.nansum(tensor)
        local_sq_sum = torch.nansum(tensor * tensor)
        local_count = torch.tensor(
            [torch.sum(~torch.isnan(tensor))], dtype=torch.long, device=tensor.device
        )

        # Stack them into one tensor so we only do one all_reduce
        packed = torch.cat(
            [local_sum.unsqueeze(0), local_sq_sum.unsqueeze(0), local_count.float()]
        )
        # e.g. packed = [local_sum, local_sq_sum, local_count]

        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        # Now packed = [global_sum, global_sq_sum, global_count]

        global_sum = packed[0]
        global_sq_sum = packed[1]
        global_count = packed[2]

        # Mean and variance on GPU
        if global_count.item() != 0:
            global_mean = global_sum / global_count
            # population variance: E[x^2] - E[x]^2
            var = (global_sq_sum / global_count) - (global_mean * global_mean)
            global_std = var.sqrt()
        else:
            global_mean = torch.tensor(torch.nan, device=global_sum.device)
            global_std = torch.tensor(torch.nan, device=global_sum.device)

    else:
        # No distributed or single rank
        global_mean = torch.nanmean(tensor)
        num_nonnan_elements = torch.sum(~torch.isnan(tensor)).item()
        if num_nonnan_elements == 0:
            global_std = torch.tensor(torch.nan, device=global_mean.device)
        elif num_nonnan_elements == 1:
            global_std = torch.tensor(
                0, dtype=global_mean.dtype, device=global_mean.device
            )
        else:
            global_std = tensor[~tensor.isnan()].std()

    # NOTE: Converting to Python float forces a sync. If you want to avoid sync
    # completely, return the GPU tensors. But typically for logging you do want floats.
    mean_val = global_mean.detach()  # .cpu().item()
    std_val = global_std.detach()  # .cpu().item()

    return mean_val, std_val


def get_sum(tensor: torch.Tensor, use_global: bool) -> torch.Tensor:
    if (
        use_global
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        local_sum = torch.nansum(tensor)

        torch.distributed.all_reduce(local_sum, op=torch.distributed.ReduceOp.SUM)
        global_sum = local_sum
    else:
        global_sum = torch.nansum(tensor)

    return global_sum.detach()
