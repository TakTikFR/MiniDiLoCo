import torch
import torch.distributed as distributed
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline


def MiniDiLoCo(
        inner_lr: float = 4e-4,
        outter_lr: float = 0.7,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        batch_size: int = 512,
        seq_length: int = 1024,
        H: int = 500,
        total_steps: int = 24000,
        betas: (float, float) = (0.9, 0.95),
        momentum: float = 0.9,
        eps: flaot = 10e-1
):
    # All these values come from the hyperparameter table of https://arxiv.org/pdf/2311.08105

    # Load custom GPT-2 with PyTorch
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path='gpt2')
    gpt2 = AutoModelForCausalLM.from_config(config)

    # Shared the data (parameters weight) to all device
    for param in gpt2.parameters():
        distributed.broadcast(param.data, src=0)

    # Setup inner and outer optimizers
    inner_optimizer = torch.optim.AdamW(gpt2.parameters(), 4e-4, betas, eps, weight_decay)
    outer_optimizer = torch.optim.SGD(gpt2.parameters(), outter_lr, momentum, nesterov=True)

    local_param = [param.data.detach().clone().to("cpu") for group in outer_optimizer.param_groups for param in group["params"]]

    scheduler = get_cosine_schedule_with_warmup(inner_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = "</s>"

    ds = load_dataset("wikitext", "wikitext-103-v1")


if __name__ == "__main__":
    MiniDiLoCo()