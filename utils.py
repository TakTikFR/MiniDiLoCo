import torch
import torch.distributed as dist

def setup(BACKEND, RANK, WORLD_SIZE):
    dist.init_process_group(backend=BACKEND, rank=RANK, world_size=WORLD_SIZE)
    torch.cuda.set_device(RANK)

def cleanup():
    dist.destroy_process_group()

def all_reduce(outer_optimizer, inner_optimizer):
    outer_params = [param for group in outer_optimizer.param_groups for param in group['params']]
    inner_params = [param for group in inner_optimizer.param_groups for param in group['params']]
    
    for outer_p, inner_p in zip(outer_params, inner_params):
        delta = outer_p.detach() - inner_p.detach()
        dist.all_reduce(tensor=delta, op=dist.ReduceOp.AVG)
        outer_p.grad = delta