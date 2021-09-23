#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/01 Thu
# TIME: 14:03:14
# DESCRIPTION: WordAvg Model
import torch
import torch.nn as nn
from data import YelpVocab
import pickle


class WordAvg(nn.Module):
    """
    sequence每个token的embedding求平均
    送到MLP输出分类结果
    """
    def __init__(self):
        super(WordAvg, self).__init__()

        vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))

        num_embeddings = len(vocab)
        embedding_dim = vocab.vectors.shape[1]
        in_features = 1024
        num_class = 2

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.fc = nn.Sequential(nn.Linear(embedding_dim, in_features),
                                nn.ReLU(),
                                nn.Linear(in_features, num_class))

    def forward(self, x):
        x = torch.mean(self.embedding(x), dim=1)
        x = self.fc(x)
        return x
