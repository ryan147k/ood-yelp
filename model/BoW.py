#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/9/24 16:05
# DESCRIPTION: Bag of Word Model
import torch.nn as nn
import pickle


class Bow(nn.Module):
    def __init__(self):
        super(Bow, self).__init__()

        vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))
        num_words = len(vocab)

        self.fc = nn.Sequential(
            nn.Linear(num_words, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        return self.fc(x)
