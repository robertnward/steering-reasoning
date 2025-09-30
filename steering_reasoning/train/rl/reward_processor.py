from abc import ABC
from enum import Enum
from typing import Optional

import torch

from steering_reasoning.train.rl.ray_workers.distributed import get_global_mean


def nanstd(values, mean, divisor):
    var = torch.nansum((values - mean) ** 2, dim=-1, keepdim=True) / (divisor - 1)
    std = var.sqrt()

    return std


class RewardProcessorType(str, Enum):
    REINFORCE = "reinforce"
    REINFORCE_WITH_BASELINE = "reinforce_with_baseline"
    RLOO = "rloo"
    GRPO = "grpo"
    GRPO_NO_STD = "grpo_no_std"


class RewardProcessor(ABC):
    def baseline_rewards(
        self, rewards: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError


class IDRewardProcessor(RewardProcessor):
    def baseline_rewards(
        self, rewards: torch.Tensor, invalid_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if invalid_mask is not None:
            rewards = rewards.clone()
            rewards[invalid_mask] = 0.0

        return rewards, {}


class MeanBaselineRewardProcessor(RewardProcessor):
    def __init__(self, mean_baseline_coef: float) -> None:
        self.mean_reward: float | None = None
        self.mean_baseline_coef = mean_baseline_coef

    @torch.no_grad()
    def update_mean_reward(self, rewards: torch.Tensor):
        global_mean_reward: float = get_global_mean(rewards)

        if self.mean_reward is None:
            self.mean_reward = global_mean_reward
        else:
            self.mean_reward = (
                self.mean_baseline_coef * self.mean_reward
                + (1 - self.mean_baseline_coef) * global_mean_reward
            )

    def baseline_rewards(
        self, rewards: torch.Tensor, invalid_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        baseline: float = self.mean_reward if self.mean_reward is not None else 0
        advantages: torch.Tensor = rewards - baseline

        if invalid_mask is not None:
            advantages[invalid_mask] = 0.0

            rewards = rewards.clone()
            rewards[invalid_mask] = torch.nan

        self.update_mean_reward(rewards=rewards)

        return advantages, {"baseline": baseline}


class RLOORewardProcessor(RewardProcessor):
    def __init__(self, num_generations: int) -> None:
        self.num_generations = num_generations

    def baseline_rewards(
        self, rewards: torch.Tensor, invalid_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if invalid_mask is None:
            rewards = rewards.reshape(-1, self.num_generations)
            baseline: torch.Tensor = (rewards.sum(-1, keepdim=True) - rewards) / (
                self.num_generations - 1
            )
            rloo_advantages: torch.Tensor = (rewards - baseline).flatten()
        else:
            rewards = rewards.clone()
            rewards[invalid_mask] = torch.nan

            rewards = rewards.reshape(-1, self.num_generations)

            num_non_nans = torch.sum(~torch.isnan(rewards), dim=-1, keepdim=True)
            baseline: torch.Tensor = (
                torch.nansum(rewards, dim=-1, keepdim=True) - rewards
            ) / (num_non_nans - 1)
            baseline[(num_non_nans == 1).tile(1, rewards.shape[1])] = 0

            rloo_advantages: torch.Tensor = (rewards - baseline).flatten()

            rloo_advantages[invalid_mask] = 0.0

        return rloo_advantages, {
            "baseline": torch.nanmean(baseline, dim=-1),
            "baseline_inner_std": nanstd(
                values=rewards,
                mean=torch.nanmean(rewards, dim=-1, keepdim=True),
                divisor=num_non_nans,
            ),
        }


class GRPORewardProcessor(RewardProcessor):
    def __init__(self, num_generations: int) -> None:
        self.num_generations = num_generations

    def baseline_rewards(
        self, rewards: torch.Tensor, invalid_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if invalid_mask is None:
            rewards = rewards.reshape(-1, self.num_generations)
            baseline = rewards.mean(-1, keepdim=True)
            std = rewards.std(-1, keepdim=True)
            std[std == 0] = 1

            grpo_advantages: torch.Tensor = ((rewards - baseline) / std).flatten()
        else:
            rewards = rewards.clone()
            rewards[invalid_mask] = torch.nan

            rewards = rewards.reshape(-1, self.num_generations)
            baseline = torch.nanmean(rewards, dim=-1, keepdim=True)

            num_non_nans = torch.sum(~torch.isnan(rewards), dim=-1, keepdim=True)
            std = nanstd(values=rewards, mean=baseline, divisor=num_non_nans)
            std[torch.logical_or(num_non_nans == 0, num_non_nans == 1)] = 0
            std_for_metrics = std.clone()
            std[std == 0] = 1

            grpo_advantages: torch.Tensor = ((rewards - baseline) / std).flatten()
            grpo_advantages[invalid_mask] = 0.0

        return grpo_advantages, {
            "baseline": baseline,
            "baseline_inner_std": std_for_metrics,
        }


class GRPONoStdRewardProcessor(RewardProcessor):
    def __init__(self, num_generations: int) -> None:
        self.num_generations = num_generations

    def baseline_rewards(
        self, rewards: torch.Tensor, invalid_mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if invalid_mask is None:
            rewards = rewards.reshape(-1, self.num_generations)
            baseline: torch.Tensor = rewards.mean(-1, keepdim=True)

            grpo_advantages: torch.Tensor = (rewards - baseline).flatten()
        else:
            rewards = rewards.clone()
            rewards[invalid_mask] = torch.nan

            rewards = rewards.reshape(-1, self.num_generations)
            baseline = torch.nanmean(rewards, dim=-1, keepdim=True)

            grpo_advantages: torch.Tensor = (rewards - baseline).flatten()
            grpo_advantages[invalid_mask] = 0.0

        return grpo_advantages, {
            "baseline": baseline,
            "baseline_inner_std": nanstd(
                values=rewards,
                mean=torch.nanmean(rewards, dim=-1, keepdim=True),
                divisor=torch.sum(~torch.isnan(rewards), dim=-1, keepdim=True),
            ),
        }
