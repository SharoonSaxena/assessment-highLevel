import torch
from transformers import AutoModel
from torch import nn


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # New layers for regression
        self.linear1 = nn.Linear(768, 1024)  # BERT output size -> 1024
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(1024, 1024)  # Intermediate layer
        self.relu2 = nn.ReLU()
        self.final_linear = nn.Linear(1024, 1)  # Output layer for single value
        self.out = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Pass input through BERT
        outputs = self.bert(input_ids, attention_mask)

        # Use the [CLS] token embedding (first token) as the sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

        # Pass through dense layers
        x = self.relu2(self.linear1(cls_embedding))  # Shape: (batch_size, 256)
        x = self.relu2(self.linear2(x))  # Shape: (batch_size, 128)
        x = self.final_linear(x)  # Shape: (batch_size, 1)
        # Apply activation to constrain output to 1-1000
        x = self.out(x) * 999 + 1  # Scale output to [1, 1000]

        return x
