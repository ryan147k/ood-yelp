#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/07 Wed
# TIME: 12:44:47
# DESCRIPTION: XLNet Model
import torch
import torch.nn as nn
from transformers import XLNetModel


class XLNetBase(nn.Module):
    def __init__(self):
        super(XLNetBase, self).__init__()

        in_features = 1024
        num_class = 2

        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')

        self.fc = nn.Sequential(nn.Linear(768, in_features),
                                nn.ReLU(),
                                nn.Linear(in_features, num_class))

    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x[0], x[1], x[2]
        output = self.xlnet(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        cls_emb = output.last_hidden_state[:, 0, :]
        x = self.fc(cls_emb)
        return x

