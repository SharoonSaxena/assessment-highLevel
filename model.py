import torch
from transformers import AutoModel
from torch import nn


class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")

        ### New layers:
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 256)
        self.final_linear = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        outputs = self.bert(ids, attention_mask=mask)

        last_hidden_state = outputs.last_hidden_state

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        linear1_output = self.linear1(last_hidden_state)
        relu_out1 = self.relu(linear1_output)
        linear2_output = self.linear2(relu_out1)
        relu_out2 = self.relu(linear2_output)
        final_output = self.final_linear(relu_out2)

        return final_output
