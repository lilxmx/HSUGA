import os
import numpy as np
import torch


class EarlyStopping():
    """Early stops training if validation performance doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint/', trace_func=print, model='checkpoint'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = os.path.join(path, "pytorch_model.bin")
        self.trace_func = trace_func

    def __call__(self, indicator, epoch, model):
        score = indicator
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'The best score is ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


class EarlyStoppingNew():
    """Early stops training with optimizer/scheduler state saving."""

    def __init__(self, patience=7, verbose=False, delta=0, path='./checkpoint/', trace_func=print, model='checkpoint'):
        if not os.path.exists(path):
            os.makedirs(path)
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = os.path.join(path, "pytorch_model.bin")
        self.trace_func = trace_func

    def __call__(self, indicator, epoch, model, optimizer=None, scheduler=None):
        score = indicator
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, optimizer, scheduler, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(score, model, optimizer, scheduler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch):
        if self.verbose:
            self.trace_func(f'The best score is ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scheduler': scheduler.state_dict() if scheduler else None
        }, self.path)
