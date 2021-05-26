import os
import gc
import platform
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers

import torch_xla
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils


def train_fn(model, dataloader, train_loss, optimizer, device, scheduler, epoch_conf):
    """Runs one epoch of training on Single or Multi-TPU setup

    Args:
        model: XLM RoBERTa model to train
        dataloader: Training Dataloader
        train_loss: Loss function for training routine
        optimizer: Optimizer
        device: xm.xla_device() type device (1 or 8 tpu cores)
        scheduler: Scheduler with conditions
        epoch_conf: List containing current epoch number and total number of epochs
    """
    model.train()
    xm.master_print(f"\n{'-'*40}\nEpoch: {epoch_conf[0]} / {epoch_conf[1]}\n{'-'*40}\n")
    xm.master_print(f"{'='*20}Training{'='*20}\n")
    for batch_idx, cache in enumerate(dataloader):
        ids = cache['input_ids'].to(device, dtype=torch.long)
        mask = cache['attention_mask'].to(device, dtype=torch.long)
        targets = cache['target'].to(device, dtype=torch.float)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(input_ids=ids, attention_mask=mask)
        
        loss = train_loss(outputs, targets)
        loss.backward()
        xm.optimizer_step(optimizer)
        
        if batch_idx % 10 == 0:
            xm.master_print(f"Batch: {batch_idx}, loss: {loss.detach().item()}")
        
        if scheduler is not None:
            scheduler.step()
    
    del loss, outputs, ids, mask, targets
    
    gc.collect()


def valid_fn(model, dataloader, valid_loss, device):
    """Runs one epoch of training on Single or Multi-TPU setup

    Args:
        model: XLM RoBERTa model to evaluate
        dataloader: Validation Dataloader
        valid_loss: Loss function for validation routine
        device: xm.xla_device() type device (1 or 8 tpu cores)
    """
    model.eval()
    total_targets, total_predictions = [], []
    
    xm.master_print(f"{'='*20}Validation{'='*20}\n")
    with torch.no_grad():
        for batch_idx, cache in enumerate(dataloader):
            ids = cache['input_ids'].to(device, dtype=torch.long)
            mask = cache['attention_mask'].to(device, dtype=torch.long)
            targets = cache['target'].to(device, dtype=torch.float)
                    
            outputs = model(input_ids=ids, attention_mask=mask)
            
            val_loss = valid_loss(outputs, targets)
            
            if batch_idx % 10 == 0:
                xm.master_print(f"Batch: {batch_idx}, val_loss: {val_loss.detach().item()}")

            total_targets.extend(targets.cpu().detach().numpy().tolist())
            total_predictions.extend(outputs.cpu().detach().numpy().tolist())
    
    del val_loss, outputs, ids, mask, targets
    gc.collect()
    
    return total_targets, total_predictions