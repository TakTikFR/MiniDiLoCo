import os
import copy
import torch
import torch.distributed as distributed
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.distributed import init_process_group
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling


def MiniDiLoCo(
        inner_lr: float = 3e-4,
        outter_lr: float = 0.7,
        warmup_steps: int = 1000,
        weight_decay: float = 0.05,
        batch_size: int = 4,
        seq_length: int = 128,
        H: int = 50,
        total_steps: int = 20,
        betas: (float, float) = (0.9, 0.95),
        momentum: float = 0.9,
        eps: float = 1e-8,
        k: int = 2
):
    # This is small values, all good values can be found in the hyperparameter table of https://arxiv.org/pdf/2311.08105

    # Load GPT-2 with PyTorch
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path='gpt2')
    gpt2 = AutoModelForCausalLM.from_config(config)

    # Shared model
    outer_model = gpt2

    # Setup inner and outer optimizers
    inner_optimizer = torch.optim.AdamW(gpt2.parameters(), inner_lr, betas, eps, weight_decay)
    outer_optimizer = torch.optim.SGD(gpt2.parameters(), outter_lr, momentum, nesterov=True)

    # Scheduler help to a significant learning rate at the beginning and decreases as it progresses
    scheduler = get_cosine_schedule_with_warmup(inner_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set for the model

    # Function that will tokenize the dataset
    def tokenize_function(data):
        outputs = tokenizer(data["text"], truncation=True, max_length=seq_length, padding="max_length")
        return outputs

    # Load and shared the datasets for the k workers
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    data_shards = []
    for i in range(k):
        sub = ds.shard(num_shards=k, index=i)
        data_shards.append(sub)

    # Create k independant workers with their own local parameters
    workers= []
    for i in range(k):
        worker_model = copy.deepcopy(gpt2)
        worker_optimizer = torch.optim.AdamW(worker_model.parameters(), inner_lr, betas, eps, weight_decay)
        worker_scheduler = get_cosine_schedule_with_warmup(worker_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        workers.append((worker_model, worker_optimizer, worker_scheduler, data_shards[i]))

    for outer_step in range(total_steps):
        local_models = []
        
        running_loss = 0.0 # TESTING
        for i, (worker_model, worker_optimizer, worker_scheduler, data) in enumerate(workers):
            data_loader = DataLoader(data, collate_fn=collate_fn , batch_size=batch_size, shuffle=True)
            batch_iterator = iter(data_loader)

            # batch → forward → loss → backward → step → reset gradients
            for inner_step in range(H):
                batch = next(batch_iterator) # Compute the batch
                outputs = worker_model(**batch) # Forward pass
                loss = outputs.loss # Loss computation
                loss.backward() # Backward propagation
                # Update parameters
                worker_optimizer.step()
                worker_scheduler.step()
                worker_optimizer.zero_grad() # Reset the gradients
                running_loss += loss.item()

            local_models.append(worker_model)

        print(f"Outer step {outer_step} avg loss: {running_loss / (k*H)}") # TESTING

        # Vérification des poids (exemple)
        with torch.no_grad():
            for p_global, p_local in zip(outer_model.parameters(), workers[0][0].parameters()):
                print("Param difference norm:", (p_global-p_local).norm().item())

        # Average the parameters
        avg_delta = []
        for param_global, *params_local in zip(outer_model.parameters(), *(m.parameters() for m in local_models)):
            deltas = [param_global.data - param_local.data for param_local in params_local]
            avg = sum(deltas) / k
            avg_delta.append(avg)

        # Applythe outer optimizer
        for param, grad in zip(outer_model.parameters(), avg_delta):
            param.grad = grad
        outer_optimizer.step()
        outer_optimizer.zero_grad()



if __name__ == "__main__":
    MiniDiLoCo()