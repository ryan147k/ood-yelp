#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/08 Thu
# TIME: 11:13:44
# DESCRIPTION: ELMo Model from allennlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules import elmo


class ELMo(nn.Module):
    def __init__(self, dropout=0.5):
        super(ELMo, self).__init__()

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        in_features = 1024
        num_class = 2

        self.elmo = elmo.Elmo(options_file, weight_file, 1, dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(1024, in_features),
                                nn.ReLU(),
                                nn.Linear(in_features, num_class))

        self._weight_init()

    def forward(self, character_ids):
        x = self.elmo(character_ids)
        x = x['elmo_representations'][0]    # (batch, length, 1024)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
