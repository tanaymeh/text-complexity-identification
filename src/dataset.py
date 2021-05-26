import torch
import transformers
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import Config

class CLRPData(Dataset):
    def __init__(self, review, target=None, is_test=False):
        self.review = review
        self.target = target
        self.is_test = is_test
        self.tokenizer = Config.TOKENIZER
        self.max_len = Config.MAX_LEN
    
    def __len__(self):
        return len(self.review)
    
    def __getitem__(self, idx):
        text = self.review[idx]
        outputs = Config.TOKENIZER.encode_plus(
            text,
            add_special_tokens=True,
            max_length=Config.MAX_LEN,
            pad_to_max_length=True
        )
        
        ids = torch.tensor(outputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(outputs['attention_mask'], dtype=torch.long)
        
        if self.is_test:
            return {
                'input_ids': ids,
                'attention_mask': mask
            }
        else:
            target = torch.tensor(self.target[idx], dtype=torch.float)
            return {
                'input_ids': ids,
                'attention_mask': mask,
                'target': target
            }