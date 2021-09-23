#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/04/07 Wed
# TIME: 16:27:17
# DESCRIPTION:
import pandas as pd
import numpy as np


def get_ranks():
    result = ['./result/wordavg.csv',
              './result/bilstm.csv',
              './result/dpcnn.csv']

    dfs = []
    for path in result:
        dfs.append(pd.read_csv(path, header=0, index_col=0))
        
    ranks = []
    for i in range(8):
        ranks.append([])
        for j in range(8):
            rank = np.argsort([df.iloc[i][j] for df in dfs])
            _rank = "{} {} {}".format(rank[0], rank[1], rank[2])
            ranks[i].append(_rank)

    ranks = pd.DataFrame(ranks)
    ranks.to_csv('./result/ranks.csv')


if __name__ == "__main__":
    get_ranks()