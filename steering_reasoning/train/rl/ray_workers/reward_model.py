from typing import Dict

import ray
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from steering_reasoning.train.rl.ray_workers.distributed_torch_ray_actor import (
    DistributedTorchRayActor,
)


class DummyRewardModel:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        logger.info(f"Vocab size in Reward Model: {self.vocab_size}")

    def init_model_from_pretrained(self, seed: int, timeout: int, rm_model):
        pass

    def eval(self):
        pass

    def forward(self, x: Dict[str, torch.Tensor]):
        reward = []
        for idx in range(len(x["input_ids"])):
            reward.append(
                x["input_ids"][idx][x["attention_mask"][idx].to(torch.bool)][-1]
            )
        # logger.debug(f"Reward: {x['input_ids'][:, -2].unsqueeze(-1).to(torch.float32)}", flush=True)
        reward = torch.tensor(reward, dtype=torch.float64, device=x["input_ids"].device)
        reward = (reward - (self.vocab_size / 2)) / self.vocab_size
        logger.debug(f"Reward: {reward}")
        return reward


@ray.remote(num_gpus=1)
class RewardModel(DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(world_size, rank, local_rank, master_addr, master_port)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()

    def init_model_from_pretrained(self, seed: int, timeout: int, rm_model):
        self._setup_distributed(seed=seed, timeout=timeout)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            rm_model,
            num_labels=1,  ##FIXME hardcoding all this
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            rm_model,
            trust_remote_code=True,  # FIXME!!!
        )

        self.model.config.pad_token_id = 151643  # FIXME
        self.tokenizer.pad_token = "<|endoftext|>"  # FIXME

        self.model.eval()

        # logger.debug(f"Reward model initialized on Node {self.node_id}, Local Rank {self.local_rank}")
        # logger.debug("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))

    def tokenize(self, text: str):
        return self.tokenizer(text, return_tensors="pt")

    def generate(self, text: str):
        tokenized_input = self.tokenize(text).to("cuda")
        return self.model(**tokenized_input)

    def eval(self):
        return self.model.eval()

    @torch.no_grad
    def forward(self, x):
        torch.cuda.empty_cache()
        x = {k: v.cuda() for k, v in x.items()}

        max_len = x["prompt_input_ids"].size(0)

        # FIXME!!!!
        eoses = (
            self.tokenizer.encode(
                "\n<|im_end|>", return_tensors="pt", add_special_tokens=False
            )
            .repeat(max_len, 1)
            .cuda()
        )
        eoses_attn_mask = torch.ones(eoses.shape, device=x["prompt_input_ids"].device)

        x["prompt_input_ids"] = torch.concat((x["prompt_input_ids"], eoses), dim=1)
        x["prompt_attention_mask"] = torch.concat(
            (x["prompt_attention_mask"], eoses_attn_mask), dim=1
        )

        position_ids = (x["prompt_attention_mask"].cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(
            x["prompt_attention_mask"].to(torch.bool) == 0, 0
        ).cuda()

        return self.model(
            input_ids=x["prompt_input_ids"],
            attention_mask=x["prompt_attention_mask"],
            position_ids=position_ids,
        ).logits
