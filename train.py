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
import torch_xla.utils.serialization as xser
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils

from src.dataset import CLRPData
from src.config import Config
from src.models import XLMModel
from src.functions import train_fn, valid_fn

modelWrap = xmp.MpModelWrapper(XLMModel(Config=Config))

def prepare_dataset():
    data = pd.read_csv(Config.FILE_NAME)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data[['excerpt', 'target']]
    train_count = 2700
    
    train_data = data[:train_count]
    valid_data = data[train_count:]
    
    train_set = CLRPData(
        review=train_data['excerpt'],
        target=train_data['target'],
        is_test=False
    )
    
    valid_set = CLRPData(
        review=valid_data['excerpt'],
        target=valid_data['target'],
        is_test=False
    )
    
    return train_set, valid_set

def _run():
    """
    Binds the entire training process together in one routine
    """
    gc.collect()
    
    xm.master_print('Starting Run...')
    
    train_dataset, valid_dataset = prepare_dataset()
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BS,
        sampler=train_sampler,
        drop_last=False,
        num_workers=8
    )
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=Config.VALID_BS,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=4
    )
    
    gc.collect()
    
    device = xm.xla_device()
    model = modelWrap.to(device)
    xm.master_print('Model Loaded')
    
    num_train_steps = int(2834 / Config.TRAIN_BS / xm.xrt_world_size())

    optimizer = transformers.AdamW([{'params': model.roberta.parameters(), 'lr': Config.LR},
                    {'params': [param for name, param in model.named_parameters() if 'roberta' not in name], 'lr': 1e-3} ], lr=LR, weight_decay=0)

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = num_train_steps * Config.EPOCHS
    )
    
    for epoch in range(Config.EPOCHS):
        train_sampler.set_epoch(epoch)
        
        para_loader = pl.ParallelLoader(train_dataloader, [device])
        xm.master_print('parallel loader created... training now')
        train_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler, epoch_conf=[epoch, Config.EPOCHS])
        
        del para_loader
        gc.collect()
        
        # using xm functionality for memory-reduced model saving
        if epoch == Config.EPOCHS-1:
            xm.master_print('saving model')
            xser.save(model.state_dict(), f"./model.bin", master_only=True)
            xm.master_print('Model Saved.')
        
        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        _, _, _ = valid_fn(para_loader.per_device_loader(device), model, device, epoch_conf=[epoch, Config.EPOCHS])

        gc.collect()
        
        del para_loader
        
# Start training processes
def _mp_fn(rank, flags):
    
    # not the cleanest way, but works
    # collect individual core outputs and save
    # can also do test inference outside training routine loading saved model
    test_preds, test_index = _run()
    np.save(f"test_preds_{rank}", test_preds)
    np.save(f"test_index_{rank}", test_index)
    return test_preds

if __name__ == '__main__':
    FLAGS={}
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')