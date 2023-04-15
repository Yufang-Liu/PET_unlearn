import copy

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, BertModel


class MultiTaskClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertModel.from_pretrained(config['model']['model_name']).cuda()
        dataset_str_list = config['data']['multi_task']
        self.classifier_list = []
        for dataset_str in dataset_str_list:
            self.classifier_list.append(
                torch.nn.Linear(config['model']['hidden_dim'],
                                config['data'][dataset_str + '_num_class'])
            )
        self.config = config

    def forward(self, batch):
        logits = self.classifier(self.model(
            input_ids=batch["src_input_ids"].cuda(),
            attention_mask=batch["src_attention_mask"].cuda(),
        )[1]).squeeze(-1)
        return logits
