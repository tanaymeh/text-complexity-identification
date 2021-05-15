import os
import gc
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import StratifiedKFold

from src.config import Config
from src.data import DistilBERTData
from src.models import TextRegressionModel
from src.trainer import Trainer

def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.003,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters, lr=Config.LR)

# Training Code
if __name__ == '__main__':
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')
    
    data = pd.read_csv(Config.FILE_NAME)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data[['excerpt', 'target']]
    
    # Do Kfolds training and cross validation
    kf = StratifiedKFold(n_splits=Config.N_SPLITS)
    nb_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, 'bins'] = pd.cut(data['target'], bins=nb_bins, labels=False)
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=data, y=data['bins'].values)):
        print(f"Fold: {fold}")
        print('-'*20)
        
        train_data = data.loc[train_idx]
        valid_data = data.loc[valid_idx]
        
        train_set = DistilBERTData(
            review = train_data['excerpt'].values,
            target = train_data['target'].values
        )

        valid_set = DistilBERTData(
            review = valid_data['excerpt'].values,
            target = valid_data['target'].values
        )

        train = DataLoader(
            train_set,
            batch_size = Config.TRAIN_BS,
            shuffle = True,
            num_workers=8
        )

        valid = DataLoader(
            valid_set,
            batch_size = Config.VALID_BS,
            shuffle = False,
            num_workers=8
        )

        model = TextRegressionModel().to(DEVICE)
        nb_train_steps = int(len(train_data) / Config.TRAIN_BS * Config.NB_EPOCHS)
        optimizer = yield_optimizer(model)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=nb_train_steps
        )

        trainer = Trainer(model, optimizer, scheduler, train, valid, DEVICE)

        best_loss = 100
        for epoch in range(1, Config.NB_EPOCHS+1):
            print(f"\n{'--'*5} EPOCH: {epoch} {'--'*5}\n")

            # Train for 1 epoch
            trainer.train_one_epoch()

            # Validate for 1 epoch
            current_loss = trainer.valid_one_epoch()

            if current_loss < best_loss:
                print(f"Saving best model in this fold: {current_loss:.4f}")
                torch.save(trainer.get_model().state_dict(), f"bert_base_uncased_fold_{fold}.pt")
                best_loss = current_loss
        
        print(f"Best RMSE in fold: {fold} was: {best_loss:.4f}")
        print(f"Final RMSE in fold: {fold} was: {current_loss:.4f}")
        
        del train_set, valid_set, train, valid, model, optimizer, scheduler, trainer, current_loss
        gc.collect()
        torch.cuda.empty_cache()