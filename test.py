#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/06/13 Sun
# TIME: 17:03:26
# DESCRIPTION: 结果记录和绘制
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd


matplotlib.use('Agg')


distence0 = [0.0, 3.302138104827827, 2.8941869178023105, 4.219576956414729, 2.8198624439130366, 3.967843922031483, 5.688415315241201, 4.250057785239702]	
distribution_rank0 = [0, 4, 2, 1, 5, 3, 7, 6]

distence1 = [0.0, 3.690717996973666, 4.2416581039667705, 5.072545168283904, 3.2765369751084537, 4.349266662031184, 3.3141567445808637, 3.187901663243603]
distribution_rank1 = [0, 7, 4, 6, 1, 2, 5, 3]

distence = [distence0[distribution_rank0[i]] + distence1[distribution_rank1[i]] for i in range(8)]
acc_wordavg = np.array([0.9394917305365067,0.8228679347759001,0.7491737446248978,0.8732340276969627,0.8521815219359121,0.6913349812507357,0.8635822418778704,0.3114924139870959])
acc_bilstm = np.array([0.9616780960064543,0.8415562934343871,0.7729367307532843,0.8987897125567322,0.756743708889023,0.735357917570499,0.8277964593664543,0.4002390956558429])
acc_attention = np.array([0.9509883017345704,0.8671324355350376,0.8371062700640867,0.9120330501571047,0.8989047130529237,0.7807260925860532,0.8937286561032096,0.47683760834021904])
acc_dpcnn = np.array([0.9713594191206132,0.8323355510843938,0.7482971439402016,0.9008495286861399,0.6124253213445175,0.7456153626259059,0.7476807934368744,0.4443311474833424])
acc_bert = np.array([0.9592577652279145,0.8637008875118808,0.8022673158250115,0.9133597113929943,0.8064238730311991,0.790630412483815,0.8351629659675246,0.48612014556706107])


def plot():
    plt.plot(distence, acc_wordavg, '-o', label='WordAvg')
    plt.plot(distence, acc_bilstm, '-o', label='BiLSTM')
    plt.plot(distence, acc_attention, '-o', label='SelfAtt')
    plt.plot(distence, acc_dpcnn, '-o', label='DPCNN')
    plt.plot(distence, acc_bert, '-o', label='Bert')
    plt.xlabel('Distribution gap')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('z.png')


    labels = ['OOD0', 'OOD1', 'OOD2', 'OOD3', 'OOD4', 'OOD5', 'OOD6', 'OOD7']

    x = np.arange(len(labels)) *  len(labels)*1 # the label locations
    width = 1  # the width of the bars
    plt.figure(figsize=(1,2))
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.set(ylim=[0.3, 1])
    ax.bar(x - width*2, acc_wordavg, width, label='WordAvg')
    ax.bar(x - width*1, acc_bilstm, width, label='BiLSTM')
    ax.bar(x, acc_attention, width, label='SelfAtt')
    ax.bar(x + width*1, acc_dpcnn, width, label='DPCNN')
    ax.bar(x + width*2, acc_bert, width, label='Bert')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acc')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.savefig('zzz.png')


def get_weighted_rank(skip):
    print(skip)
    acc = np.vstack((acc_wordavg, acc_bilstm, acc_attention, acc_dpcnn, acc_bert))
    print(np.argsort(acc[:, skip]))
    acc_weighted = np.zeros(5)
    dist = 0
    for i in range(1, 8):
        if i != skip:
            acc_weighted += acc[:, i] * distence[i]
            dist += distence[i]
    acc_weighted = acc_weighted / dist
    acc_avg = (acc[:, 0] + acc_weighted) / 2
    print(np.argsort(acc_avg))


def get_rank():
    acc = np.vstack((acc_wordavg, acc_bilstm, acc_attention, acc_dpcnn, acc_bert))
    # acc = 1 - acc
    _rank = np.argsort(acc, axis=0)
    rank = np.zeros_like(_rank)
    for i in range(_rank.shape[1]):
        rank_c = _rank[:, i]
        for j in range(len(rank_c)):
            rank[rank_c[j], i] = j
    rank += 1 
    return rank


def rank_distance(rank1, rank2):
    dis = 0
    for r1, r2 in zip(rank1, rank2):
        dis += abs(r1 - r2)
    return dis


def get_drop(acc):
    res = []
    for i in range(len(acc)):
        res.append(acc[0] - acc[i])
    return res


# acc_wordavg = get_drop(acc_wordavg)
# acc_bilstm = get_drop(acc_bilstm)
# acc_attention = get_drop(acc_attention)
# acc_dpcnn = get_drop(acc_dpcnn)
# acc_bert = get_drop(acc_bert)

rank = get_rank()

for i in range(0, len(rank[0])):
    print(rank_distance(rank[:, 0].tolist(), rank[:, i].tolist()))

# rank = pd.DataFrame(rank, index=['WordAvg','BiLSTM','SelfAtt','DPCNN','Bert'], columns=['OOD0','OOD1','OOD2','OOD3','OOD4','OOD5','OOD6','OOD7'])
# rank.to_csv('tmp.csv')

acc = np.vstack((acc_wordavg, acc_bilstm, acc_attention, acc_dpcnn, acc_bert))
acc = pd.DataFrame(acc, index=['WordAvg','BiLSTM','SelfAtt','DPCNN','Bert'], columns=distence)
acc.to_csv('acc.csv')

# plot()