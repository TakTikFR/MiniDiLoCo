import copy
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling

def apply_sync_algorithm(method_name: str, global_params, local_params_list):
    aggregated_update = []
    nb_workers = len(local_params_list)

    if method_name == "basic":
        # Compute the average of the deltas
        for param_global, *params_local in zip(global_params, local_params_list):
            deltas = [param_global.data - param_local.data for param_local in params_local]
            avg = sum(deltas) / nb_workers                                                                      # Δ(t) ← average of ( θ(t-1) - θᵢ(t) ) for i=1...k
            aggregated_update.append(avg)
    elif method_name == "allreduce":
        # Applt the sum of the deltas (AllReduce)
        for param_global, *params_local in zip(global_params, local_params_list):
            deltas = [param_global.data - param_local.data for param_local in params_local]
            aggregated_update.append(sum(deltas))
    else:
        raise ValueError("Wrong Method name, should be 'basic' or 'allreduce'")
    return aggregated_update


def MiniDiLoCo(
        inner_lr: float = 2e-4,
        outter_lr: float = 0.7,
        warmup_steps: int = 100,
        weight_decay: float = 0.05,
        batch_size: int = 4,
        seq_length: int = 64,
        H: int = 10,
        total_steps: int = 10,
        betas: (float, float) = (0.9, 0.95),
        momentum: float = 0.9,
        eps: float = 1e-8,
        k: int = 2
):
    """
    Sequential simulation of the DiLoCo distributed training algorithm.

    This function lets you experiment with DiLoCo's core logic (local updates, parameter aggregation and synchronization)
    in a simple, single-process and non-parallel setting. It is intended for conceptual and learning purposes, not true distributed training.
    """

    # This is small values, all good values can be found in the hyperparameter table of the paper : https://arxiv.org/pdf/2311.08105

    # Load GPT-2 model
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path='gpt2')
    gpt2 = AutoModelForCausalLM.from_config(config)

    # Shared model
    outer_model = gpt2

    # Setup inner and outer optimizers
    inner_optimizer = torch.optim.AdamW(gpt2.parameters(), inner_lr, betas, eps, weight_decay) # AdamW
    outer_optimizer = torch.optim.SGD(gpt2.parameters(), outter_lr, momentum, nesterov=True) # Nesterov

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

    # Create k independant workers with their own local parameters, optimizers and scheduler
    workers= []
    for i in range(k):
        worker_model = copy.deepcopy(gpt2)
        worker_optimizer = torch.optim.AdamW(worker_model.parameters(), inner_lr, betas, eps, weight_decay)
        worker_scheduler = get_cosine_schedule_with_warmup(worker_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        workers.append((worker_model, worker_optimizer, worker_scheduler, data_shards[i]))

    for outer_step in range(total_steps):                                                               # for t = 1...T do
        local_models = []
        
        running_loss = 0.0 # Cumulative average loss across all inner steps and workers
        for i, (worker_model, worker_optimizer, worker_scheduler, data) in enumerate(workers):              # for i = 1...k do
            data_loader = DataLoader(data, collate_fn=collate_fn , batch_size=batch_size, shuffle=True)         # θᵢ(t) ← θᵢ(t-1)
            batch_iterator = iter(data_loader)

            # batch → forward → loss → backward → step → reset gradients
            for inner_step in range(H):                                                                         # for h = 1...H do
                # Compute the batch
                # Handle the case where we reach the end of the shard
                try:
                    batch = next(batch_iterator)                                                                    # x ∼ Dᵢ 
                except StopIteration:
                    batch_iterator = iter(data_loader)
                    batch = next(batch_iterator)

                outputs = worker_model(**batch) # Forward pass

                loss = outputs.loss # Loss computation                                                              # L ← f( x, θᵢ(t) )

                loss.backward() # Backward propagation

                # Update parameters
                worker_optimizer.step()                                                                             # θᵢ(t) ← InnerOpt( θᵢ(t), ∇L )
                worker_scheduler.step()

                worker_optimizer.zero_grad() # Reset the gradients

                running_loss += loss.item()
                                                                                                                # end for
            local_models.append(worker_model)
                                                                                                            # end for

        # Display the average loss at each outer step
        print(f"Outer step {outer_step} avg loss: {running_loss / (k*H)}") # For testing

        # Show the model parameters, to check that the parameters has been updated
        with torch.no_grad():
            for p_global, p_local in zip(outer_model.parameters(), workers[0][0].parameters()):
                print("Param difference norm:", (p_global-p_local).norm().item())

        
        # Average the parameters
        aggregated_update = []
        for param_global, *params_local in zip(outer_model.parameters(), *(m.parameters() for m in local_models)):
            deltas = [param_global.data - param_local.data for param_local in params_local]
            avg = sum(deltas) / k                                                                       # Δ(t) ← average of ( θ(t-1) - θᵢ(t) ) for i=1...k
            avg_delta.append(avg)
        '''
            
        aggregated_update = apply_sync_algorithm("allreduce", list(outer_model.parameters()), [list(m.parameters()) for m in local_models])
        '''
        # Apply the outer optimizer
        for param, grad in zip(outer_model.parameters(), aggregated_update):
            param.grad = grad                                                                           # θ(t) ← OuterOpt( θ(t-1), Δ(t) )
        outer_optimizer.step()
        outer_optimizer.zero_grad()

                                                                                                    # end for

if __name__ == "__main__":
    MiniDiLoCo()