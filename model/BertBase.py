#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/01 Thu
# TIME: 14:06:40
# DESCRIPTION: BertBase Model
import torch
import torch.nn as nn
from transformers import BertModel


class BertBase(nn.Module):
    def __init__(self):
        super(BertBase, self).__init__()

        in_features = 1024
        num_class = 2

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Sequential(nn.Linear(768, in_features),
                                nn.ReLU(),
                                nn.Linear(in_features, num_class))

    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x[0], x[1], x[2]
        output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = output.last_hidden_state[:, 0, :]
        x = self.fc(cls_emb)
        return x
