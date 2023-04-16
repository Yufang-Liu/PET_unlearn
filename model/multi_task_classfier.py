import copy

import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class MultiTaskClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertModel.from_pretrained(config['model']['model_name']).cuda()
        dataset_str_list = config['data']['multi_task']
        self.classifier_list = []
        for dataset_str in dataset_str_list:
            self.classifier_list.append(
                torch.nn.Linear(config['model']['hidden_dim'],
                                config['data'][dataset_str + '_num_class'],
                                device=config['options']['device'])
            )
        self.config = config

    def forward(self, batch, train=True):
        task_id = batch['task_id']
        loss = 0
        target_list, prediction_list = [], []
        for idx, classifier in enumerate(self.classifier_list):
            task_indice = task_id == idx
            task_indice.to(self.config['options']['device'])
            target = batch['labels'][task_indice]
            outputs = classifier(self.model(
                input_ids=batch["input_ids"][task_indice],
                attention_mask=batch["attention_mask"][task_indice],
            )[1]).squeeze(-1)
            if train:
                loss += F.cross_entropy(outputs, target)
            else:
                prediction = torch.argmax(outputs, dim=-1)
                prediction_list.append(prediction)
                target_list.append(target)
        if train:
            return loss
        else:
            prediction = torch.cat(prediction_list)
            targets = torch.cat(target_list)
            return prediction, targets
