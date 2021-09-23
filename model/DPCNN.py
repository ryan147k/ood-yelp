#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/01 Thu
# TIME: 14:17:11
# DESCRIPTION: Deep Pyramid CNN Model
import torch
import torch.nn as nn
from data import YelpVocab
import pickle


class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self):
        super(DPCNN, self).__init__()

        vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))

        num_embeddings = len(vocab)
        embedding_dim = vocab.vectors.shape[1]
        num_class = 2

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(in_channels=1,
                                               out_channels=self.channel_size,
                                               kernel_size=(3, embedding_dim),
                                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=self.channel_size, 
                               out_channels=self.channel_size,
                               kernel_size=(3, 1), 
                               stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.channel_size, num_class)

    def forward(self, x):
        batch = x.shape[0]

        x = self.embedding(x).unsqueeze(dim=1)  # dim 1 添加一维 channel_size
        # Region embedding
        x = self.conv_region_embedding(x)       # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            if x.shape[2] == 3:
                x = self.padding_pool(x)
            x = self._block(x)

        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)

        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = torch.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = torch.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
