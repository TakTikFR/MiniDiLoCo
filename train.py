import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import all_reduce

class Trainer:
    """ Generic distributed PyTorch trainer """

    def __init__(self, rank, world_size, model, dataloader):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.model = DDP(model.to(self.device), device_ids=[rank])
        self.dataloader = dataloader


    def train(self, strategy, total_steps):
        """ Distributed outer training loop.

        Args:
            strategy (Strategy): Distributed strategy chooses who performs the inner loop
            total_steps (Int): Number of outer loops
        """        

        strategy._init_node(self.model, self.rank, self.world_size, total_steps)
        
        self.model.train()

        for outer_step in range(total_steps):
            self.dataloader.sampler.set_epoch(outer_step)

            running_loss = 0.0
            for inner_step, batch in enumerate(self.dataloader):
                if inner_step >= strategy.H:
                    break

                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Inner loop
                loss = strategy.step(batch)
                running_loss += loss

            # Parameters synchronisation
            all_reduce(strategy.outer_opt, strategy.inner_opt)

            strategy.outer_opt.zero_grad()
            strategy.outer_opt.step()

            print(f"Rank: {self.rank} - Outer step: {outer_step} - Average loss: {running_loss / strategy.H}")