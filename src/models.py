import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class TextRegressionModel(nn.Module):
    
    def __init__(self, model_name, dropout_p=0.1):
        super(TextRegressionModel, self).__init__()
        
        self.model = AutoModel.from_pretrained(model_name)
        self.features = nn.Linear(768, 768)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(768, 1)
        
    def forward(self, input_ids, attention_mask):
        
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        output = F.relu(self.features(output.last_hidden_state[:, 0]))
        output = self.dropout(output)
        output = self.out(output)
        return output