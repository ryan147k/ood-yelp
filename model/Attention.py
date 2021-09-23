#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/14 Wed
# TIME: 12:37:55
# DESCRIPTION: Attention
import torch
import torch.nn as nn
from data import YelpVocab
import pickle


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

        vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))

        num_embeddings = len(vocab)
        embedding_dim = vocab.vectors.shape[1]
        in_features = 1024
        num_class = 2

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        self.Q = nn.Linear(embedding_dim, 128)
        self.K = nn.Linear(embedding_dim, 128)
        self.V = nn.Linear(embedding_dim, 128)
        
        self.self_attention = nn.MultiheadAttention(128, num_heads=4)

        self.fc = nn.Sequential(nn.Linear(128, in_features),
                                nn.ReLU(),
                                nn.Linear(in_features, num_class))
    
    def forward(self, x):
        x = self.embedding(x)
        query = self.Q(x).permute(1, 0, 2)
        key = self.K(x).permute(1, 0, 2)
        value = self.V(x).permute(1, 0, 2)
        x, _ = self.self_attention(query, key, value)
        x = x.permute(1, 0, 2)      # (batch, seq, emb)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
