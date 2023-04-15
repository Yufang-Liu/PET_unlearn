

import copy

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class SingleTaskClassifier(torch.nn.Module):
    def __init__(self, config, num_label):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(config['model']['model_name'],
                                                                        return_dict=True,
                                                                        num_labels=num_label)

    def forward(self, batch):
        pass
