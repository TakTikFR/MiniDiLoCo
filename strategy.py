import torch
from copy import deepcopy
from torch.optim import AdamW, SGD
from transformers import get_cosine_schedule_with_warmup

from abc import ABC, abstractmethod
from typing import override

class Strategy(ABC):
    """ Base class for training strategy. """

    def _init_node(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def step(self, batch):
        pass

class Diloco(Strategy):
    """     DiLoCo PyTorch strategy.     """
    """ https://arxiv.org/pdf/2311.08105 """

    def __init__(self,
        inner_lr: float = 4e-4,
        outer_lr: float = 0.7,
        warmup_steps: int = 1000,
        weight_decay: float = 0.1,
        H: int = 500,
        betas: (float, float) = (0.9, 0.95),
        momentum: float = 0.9,
        eps: float = 10e-1,
    ):

        """ The parameters come from Table 5 of the paper. """

        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.H = H
        self.betas = betas
        self.momentum = momentum
        self.eps = eps

    @override
    def _init_node(self, model, rank, world_size, total_steps=100):
        """ Initialize the optimizers and scheduler of DiLoCo """

        super()._init_node(model, rank, world_size)

        device = torch.device(f"cuda:{rank}")

        local_params = model.parameters()
        global_params = [p.clone().detach().requires_grad_(True) for p in model.parameters()]

        self.inner_opt = AdamW(
            local_params,
            self.inner_lr,
            self.betas,
            self.eps,
            self.weight_decay
        )

        self.outer_opt = SGD(
            global_params,
            self.outer_lr,
            self.momentum,
            nesterov=True
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.inner_opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps * self.H
        )

        print(f"Rank {self.rank}: DiLoCo H={self.H}")
        
    def step(self, batch):
        """ Inner loop of the DiLoCo strategy """

        self.inner_opt.zero_grad()
        
        outputs = self.model(**batch)

        loss = outputs.loss
        loss.backward()

        self.inner_opt.step()
        self.scheduler.step()

        return loss.item()