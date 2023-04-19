import torch
import torch.nn as nn
from transformers import BertModel, AutoModelForSequenceClassification
import torch.nn.functional as F


class MultiTaskClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = BertModel.from_pretrained(config['model']['model_name']).cuda()
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        dataset_str_list = config['data']['multi_task']
        self.classifier_list = []
        for dataset_str in dataset_str_list:
            self.classifier_list.append(
                torch.nn.Linear(config['model']['hidden_dim'],
                                config['data'][dataset_str + '_num_class'],
                                device=config['options']['device'])
            )
        self.config = config

    def get_prefix_seq_cls_model(self, task_id):
        dataset_str = self.config['data']['multi_task'][task_id]
        seq_cls_model = AutoModelForSequenceClassification.\
            from_pretrained(self.config['model']['model_name'],
                            return_dict=True,
                            num_labels=self.config['data'][dataset_str + '_num_class'],
                            cache_dir=self.config['model']['cache_dir'])
        state_dict = seq_cls_model.state_dict()
        for (ori_k, ori_v), (new_k, new_v) in zip(state_dict.items(), self.model.state_dict().items()):
            assert ori_k == 'bert.' + new_k
            state_dict[ori_k] = self.model.state_dict()[new_k]

        state_dict['classifier.weight'] = \
            self.classifier_list[task_id].state_dict()['weight']
        state_dict['classifier.bias'] = \
            self.classifier_list[task_id].state_dict()['bias']
        seq_cls_model.load_state_dict(state_dict)
        return seq_cls_model

    def forward(self, batch, train=True):
        task_id = batch['task_id']
        loss = 0
        target_list, prediction_list = [], []
        for idx, classifier in enumerate(self.classifier_list):
            task_indice = task_id == idx
            target = batch['labels'][task_indice]
            outputs = classifier(self.dropout(self.model(
                input_ids=batch["input_ids"][task_indice],
                attention_mask=batch["attention_mask"][task_indice],
            )[1])).squeeze(-1)
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
