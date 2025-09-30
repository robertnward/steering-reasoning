def register_patch_head():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "steering_reasoning.models.head_patched.qwen.vllm:Qwen2ForCausalLM_CustomLast",
    )

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "steering_reasoning.models.head_patched.llama.vllm:LlamaForCausalLM_CustomLast",
    )


def register_add_place():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        "steering_reasoning.models.steering_place.vllm:SteeredQwen2ForCausalLM",
    )

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "steering_reasoning.models.steering_place.vllm:SteeredLlamaForCausalLM",
    )


def register_patch_head_path():
    from vllm import ModelRegistry

    # ModelRegistry.register_model(
    #     "Qwen2ForCausalLM",
    #     "steering_reasoning.models.head_patched_path.qwen.vllm:Qwen2ForCausalLM_CustomLast",
    # )

    ModelRegistry.register_model(
        "LlamaForCausalLM",
        "steering_reasoning.models.head_patched_path.llama.vllm:LlamaForCausalLM_CustomLast",
    )
