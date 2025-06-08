import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
import time, random as _rand
import math, os, argparse, random

class DecayingCosineAnnealingWarmRestartsWithShuffle(LRScheduler):
    def __init__(
        self, optimizer, T_0, T_mult=1, decay_factor=0.9,
        eta_min=0, last_epoch=-1
    ):
        self.T_0, self.T_mult = T_0, T_mult  # T_0 is now in terms of iterations
        self.decay_factor, self.eta_min = decay_factor, eta_min

        self.T_i = T_0
        self.T_cur = 0
        self.cycle = 0 
        self.base_max_lrs = [g['lr'] for g in optimizer.param_groups]

        self.original_schedule = []
        self.all_original_lrs = []

        super().__init__(optimizer, last_epoch)

    def _generate_schedule(self):
        decayed_max_lrs = [
            base_lr * (self.decay_factor ** self.cycle)
            for base_lr in self.base_max_lrs
        ]

        self.original_schedule = []

        for max_lr in decayed_max_lrs:
            lrs = [
                self.eta_min + 0.5 * (max_lr - self.eta_min) *
                (1 + math.cos(math.pi * t / self.T_i))
                for t in range(self.T_i)
            ]
            self.original_schedule.append(lrs.copy())
            self.all_original_lrs.extend(lrs)

    def get_lr(self):
        self._generate_schedule()
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = 0
            self.T_i *= self.T_mult
            self._generate_schedule()
        return [lrs[self.T_cur] for lrs in self.original_schedule]

    def step(self, epoch=None):
        self._generate_schedule()
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = 0
            self.T_i *= self.T_mult
            self._generate_schedule()
        else:
            self.T_cur += 1
        self.last_epoch += 1
        new_lrs = self.get_lr()
        for pg, lr in zip(self.optimizer.param_groups, new_lrs):
            pg['lr'] = lr
        self._last_lr = new_lrs


class CustomReduceOnPlateau(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='max', factor=0.5, patience=5, 
                 threshold=1e-4, threshold_mode='rel', cooldown=0, 
                 min_lr=0, eps=1e-8, verbose=False, injection_prob=0.2, 
                 injection_mode='fixed', injection_factor=1.5):

        super().__init__(optimizer, mode, factor, patience, threshold, 
                         threshold_mode, cooldown, min_lr, eps, verbose)
        
        # Store custom attributes specific to this class
        self.injection_prob = injection_prob
        self.injection_factor = injection_factor
        self.injection_mode = injection_mode  # 'fixed' or 'long_tail'
        self.original_lrs = None  # Store original learning rates during injection
        self.min_lr = min_lr  # Store minimum learning rate

        # Predefine a long-tail distribution for injection factors
        self.injection_distribution = self._generate_long_tail_distribution()

    def _generate_long_tail_distribution(self):
        """Generate a long-tail distribution for injection factors."""
        distribution = [1.0] * 80 + [random.uniform(1.0, 3.0) for _ in range(20)]
        random.shuffle(distribution)
        return distribution

    def restore_original_lr(self):
        """Restore the original learning rates after injection."""
        if self.original_lrs is not None:
            for param_group, original_lr in zip(self.optimizer.param_groups, self.original_lrs):
                param_group['lr'] = original_lr
            self.original_lrs = None

    def step(self, metrics, epoch=None):
        # Restore original learning rates if they were modified
        ori_num_of_bad_epochs = self.num_bad_epochs
        self.restore_original_lr()
 

        # Perform the regular ReduceLROnPlateau step
        super().step(metrics, epoch)


        # Probabilistically inject a large learning rate
        if self.injection_mode == 'fixed':
            if _rand.random() < self.injection_prob:
                self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
                injection_factor = self.injection_factor
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * injection_factor
                self.num_bad_epochs = ori_num_of_bad_epochs 
    
            return self.original_lrs
        else:
            self.original_lrs = [pg['lr'] for pg in self.optimizer.param_groups]  # Save original learning rates
            injection_factor = random.choice(self.injection_distribution)  # Sample from the predefined distribution
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * injection_factor
            if injection_factor > 1.0:
                self.num_bad_epochs = ori_num_of_bad_epochs 
            return self.original_lrs