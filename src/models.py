import torch
import transformers
import torch.nn as nn


class XLMModel(nn.Module):
    def __init__(self, Config):
        super(XLMModel, self).__init__()
        self.backbone = transformers.XLMRobertaModel.from_pretrained(
            Config.MODEL_NAME, 
            num_labels=1,
            output_hidden_states=False
        )
        
        self.drop = nn.Dropout(0.3)
        self.ln = nn.LayerNorm(1024)
        self.head = nn.Linear(1024, 1)
        
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None, input_embeds=None):
        o1, _ = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            input_embeds=input_embeds
        )
        
        x = torch.mean(o1, 1)
        x = self.ln(x)
        x = self.drop(x)
        out = self.head(x)
        
        return out