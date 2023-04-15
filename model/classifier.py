import copy

import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, BertModel


class Classifier(torch.nn.Module):
    def __init__(self, config, unlearn=False):
        super().__init__()
        self.model = BertModel.from_pretrained(config['model']['model_name']).cuda()
        num_class = config['model']['num_class']
        self.classifier = torch.nn.Linear(config['model']['hidden_dim'], num_class).cuda()
        self.config = config
        self.unlearn = unlearn
        self.hidden_size = self.model.config.hidden_size
        self.init_range = self.model.config.initializer_range

    def forward(self, batch):
        logits = self.classifier(self.model(
            input_ids=batch["src_input_ids"].cuda(),
            attention_mask=batch["src_attention_mask"].cuda(),
        )[1]).squeeze(-1)
        return logits

    def set_prefix(self, prefix_len=10, prefix=None):
        self.prefix_list = [[nn.Parameter(torch.randn(prefix_len, self.hidden_size, device="cuda")),
                             nn.Parameter(torch.randn(prefix_len, self.hidden_size, device="cuda"))]
                            for _ in range(12)]
        if prefix:
            for par, old in zip(self.prefix_list, prefix):
                par[0] = copy.deepcopy(old[0])
                par[1] = copy.deepcopy(old[1])
        else:
            for par in self.prefix_list:
                for exp in par:
                    exp.data.normal_(mean=0.0, std=self.init_range)
        if self.unlearn:
            self.prefix_list_new = [
                [nn.Parameter(torch.randn(prefix_len, self.hidden_size, device="cuda")),
                 nn.Parameter(torch.randn(prefix_len, self.hidden_size, device="cuda"))]
                for _ in range(12)]
            for par in self.prefix_list_new:
                for exp in par:
                    exp.data.normal_(mean=0.0, std=self.init_range)
            for i in range(12):
                self.prefix_list[i].extend(self.prefix_list_new[i])
        self.model.set_prefix(self.prefix_list)

    def set_optimizer(self, lr):
        if not self.unlearn:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        exp for p in self.prefix_list for exp in p
                    ]
                },
                {
                    "params": [
                        p
                        for n, p in self.classifier.named_parameters()
                    ],
                    "lr": 3e-4
                }
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for exp in self.prefix_list_new for p in exp
                    ],
                },
                {
                    "params": [
                        p
                        for n, p in self.classifier.named_parameters()
                    ],
                    "lr": 3e-4
                }
            ]

        self.optim = torch.optim.AdamW(
            optimizer_grouped_parameters,
            #self.parameters(),
            lr=lr,
            weight_decay=self.config['optim']['weight_decay'],
        )

        self.sched = get_linear_schedule_with_warmup(
            self.optim,
            num_warmup_steps=self.config['optim']['warmup_updates'],
            num_training_steps=self.config['optim']['total_num_updates'],
        )