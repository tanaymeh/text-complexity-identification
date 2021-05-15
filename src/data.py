import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import Config

class BERTDataset(Dataset):
    def __init__(self, review, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, idx):
        review = str(self.review[idx])
        review = ' '.join(review.split())
        
        inputs = self.tokenizer.encode_plus(
            review,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )        
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long)
        
        if self.is_test:
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
            }
        else:    
            targets = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'ids': ids,
                'mask': mask,
                'token_type_ids': token_type_ids,
                'targets': targets
            }
            
class DistilBERTData(Dataset):
    def __init__(self, text, target=None, is_test=False):
        super(DistilBERTData, self).__init__()
        self.text = text
        self.target = target
        self.is_test = is_test
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())
        
        inputs = Config.tokenizer(text, 
                                  truncation=True, 
                                  padding=True, 
                                  return_tensors='pt', 
                                  add_special_tokens=True
                                )
        
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        
        if self.is_test:
            return {'inputs': input_ids,
                    'attention_mask': attention_mask
                    }
        else:
            return {'inputs': input_ids,
                    'attention_mask': attention_mask,
                    'targets': torch.tensor(self.target[index], dtype=torch.float)
                    }
        