#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/04/01 Thu
# TIME: 14:05:17
# DESCRIPTION: BiLSTM Model
import torch
import torch.nn as nn
from data import YelpVocab
import pickle


class BiLSTM(nn.Module):
    def __init__(self, dropout=0.5):
        super(BiLSTM, self).__init__()

        vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))

        num_embeddings = len(vocab)
        embedding_dim = vocab.vectors.shape[1]
        hidden_size = 256
        in_features = 1024
        num_class = 2

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=2,
                              batch_first=True,
                              bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(nn.Linear(hidden_size * 2, in_features),
                                nn.ReLU(),
                                nn.Linear(in_features, num_class))

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.bilstm(x)
        x = torch.mean(output, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
