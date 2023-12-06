"""
Module providing utility functions for training. 
Includes functions for adjusting learning rates, 
saving training checkpoints, and saving temporary 
model checkpoints during training.

Author: 
"""

import os
import json
import torch

from training.arguments import get_train_args

args = get_train_args()
LOG_FATH = args.training_logs

def adjust_learning_rate(optimizer, lr_decay):
    """
    Adjust the learning rate of the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        lr_decay (float): The factor by which the learning rate will be multiplied.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

def save_checkpoint(state, info_json, dirname) -> None:
    """
    Save a training checkpoint.

    Args:
        state (dict): Model state dictionary to be saved.
        info_json (dict): Additional information to be saved in a JSON file.
        dirname (str): Directory path where the checkpoint will be saved.
    """
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    with open(os.path.join(dirname, 'info.json'), 'w', encoding='utf-8') as f:
        json.dump(info_json, f, indent=4)
    torch.save(state, os.path.join(dirname, 'model.ckpt'))

def save_spare(name: str, epoch, model) -> None:
    """
    Save a temporary checkpoint during training.

    Args:
        name (str): Name or identifier for the saved checkpoint.
        epoch (int): Current training epoch.
        model (torch.nn.Module): The PyTorch model whose state will be saved.
    """
    save = os.path.join(LOG_FATH, f'{epoch}_{name}.pth')
    torch.save(model.state_dict(), save)
