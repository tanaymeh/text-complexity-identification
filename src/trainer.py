import gc
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

from .config import Config

class Trainer:
    def __init__(
        self, 
        model, 
        optimizer, 
        scheduler, 
        train_dataloader, 
        valid_dataloader,
        device
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.loss_fn = self.yield_loss
        self.device = device
        
    def yield_loss(self, outputs, targets):
        """
        This is the loss function for this task
        """
        return torch.sqrt(nn.MSELoss()(outputs, targets))
    
    def train_one_epoch(self):
        """
        This function trains the model for 1 epoch through all batches
        """
        prog_bar = tqdm(enumerate(self.train_data), total=len(self.train_data))
        self.model.train()
        with autocast():
            for idx, inputs in prog_bar:
                ids = inputs['inputs'].to(self.device, dtype=torch.long)
                mask = inputs['attention_mask'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(input_ids=ids, attention_mask=mask)           

                loss = self.loss_fn(outputs.squeeze(1), targets)
                prog_bar.set_description('loss: {:.2f}'.format(loss.item()))

                Config.scaler.scale(loss).backward()
                Config.scaler.step(self.optimizer)
                Config.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
    
    def valid_one_epoch(self):
        """
        This function validates the model for one epoch through all batches of the valid dataset
        It also returns the validation Root mean squared error for assesing model performance.
        """
        prog_bar = tqdm(enumerate(self.valid_data), total=len(self.valid_data))
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for idx, inputs in prog_bar:
                ids = inputs['inputs'].to(self.device, dtype=torch.long)
                mask = inputs['attention_mask'].to(self.device, dtype=torch.long)
                targets = inputs['targets'].to(self.device, dtype=torch.float)

                outputs = self.model(input_ids=ids, attention_mask=mask)
                all_targets.extend(targets.cpu().detach().numpy().tolist())
                all_predictions.extend(outputs.cpu().detach().numpy().tolist())

        val_rmse_loss = np.sqrt(mean_squared_error(all_targets, all_predictions))
        print('Validation RMSE: {:.2f}'.format(val_rmse_loss))
        
        return val_rmse_loss
    
    def get_model(self):
        return self.model