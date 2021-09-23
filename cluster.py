#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/9/20 17:43
# DESCRIPTION:  1. 获取Yelp数据集的embedding表示(bert的cls),并输出到文件
#               2. KMeans对PCA后的Yelp embedding进行聚类,并将KMeans参数输出到文件
#               3. 统计聚类结果(每个类的数量)
#               4. 从每个类别采样出10000条数据,保证正负样本均衡,并写入到文件中(yelp_clustered_csv)
#               5. 选择聚类中心模长最大的作为基类,计算其他类与基类的NI值
#               6. TSNE对采样后的Yelp进行可视化
import torchtext
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from collections import Counter
import json
import csv
import os
import shutil


# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _get_yelp_text_list(positive, root='./dataset'):
    dataset = torchtext.datasets.YelpReviewPolarity(root=root, split='train')
    text_list = []

    _label = 2 if positive else 1
    for label, text in dataset:
        if label == _label:
            text_list.append(text)
    return text_list


class YelpDataset(Dataset):
    """
    Yelp Dataset, 返回BertTokenizer的编码
    """
    def __init__(self, positive=True):
        """
        positive: 决定是正类还是负类
        """
        super(YelpDataset, self).__init__()
        yelp_text_list = _get_yelp_text_list(positive)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.input = tokenizer(yelp_text_list,
                               padding=True,
                               truncation=True,  # 截断成512
                               return_tensors='pt')

    def __getitem__(self, idx):
        input_ids = self.input.input_ids[idx]
        token_type_ids = self.input.token_type_ids[idx]
        attention_mask = self.input.attention_mask[idx]

        return input_ids, token_type_ids, attention_mask

    def __len__(self):
        return len(self.input.input_ids)


def cls_emb2pkl(positive: bool):
    """
    将Yelp数据集的Bert cls向量输出到文件
    :param positive:
    :return:
    """
    yelp_dataset = YelpDataset(positive=positive)
    yelp_dataloader = DataLoader(yelp_dataset, batch_size=256)

    model = BertModel.from_pretrained('bert-base-uncased').to(device)

    with torch.no_grad():
        x = None
        for batch in tqdm(yelp_dataloader):
            input_ids = batch[0].to(device)
            token_type_ids = batch[1].to(device)
            attention_mask = batch[2].to(device)

            output = model.forward(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)
            sentence_emb = output.last_hidden_state[:, 0, :].cpu()
            x = sentence_emb if x is None else torch.cat((x, sentence_emb), dim=0)

    name = 'pos' if positive == 'pos' else 'neg'
    pickle.dump(x, open('./count/cls_emb_{}.pkl'.format(name), 'wb'))


def _pca(positive, n_components=40):
    """
    PCA降维
    :param positive:
    :param n_components:
    :return:
    """
    emb_path = './count/cls_emb_pos.pkl' if positive else './count/cls_emb_neg.pkl'
    x = pickle.load(open(emb_path, 'rb'))
    x = x.numpy()
    x = PCA(n_components=n_components, random_state=3).fit_transform(x)
    print('PCA FINISH.')
    return x


def kmeans_estimator2pkl():
    """
    对降维后的数据进行聚类，将KMeans参数输出到文件
    :return:
    """
    emb_pos = _pca(True)
    emb_neg = _pca(False)
    pos_estimator = KMeans(n_clusters=8).fit(emb_pos)
    neg_estimator = KMeans(n_clusters=8).fit(emb_neg)

    pickle.dump(pos_estimator, open('./count/estimator_pos.pkl', 'wb'))
    pickle.dump(neg_estimator, open('./count/estimator_neg.pkl', 'wb'))


def _cluster_count(positive):
    name = 'pos' if positive else 'neg'
    estimator = pickle.load(open('./count/estimator_{}.pkl'.format(name), 'rb'))
    count = Counter([str(label) for label in estimator.labels_])
    return count


def cluster_count_res2file():
    pos = _cluster_count(True)
    neg = _cluster_count(False)

    with open('./count/cluster_count.txt', 'w') as f:
        f.write("POS\n")
        f.write(json.dumps(pos, indent=4))
        f.write("\nNEG\n")
        f.write(json.dumps(neg, indent=4))


def _get_index_list(positive):
    """
    从聚类后的每个类中选10000条样本出来, 返回样本index列表
    :param positive:
    :return:
    """
    num_samples = 10000
    random.seed(2)

    name = 'pos' if positive else 'neg'
    estimator = pickle.load(open('./count/estimator_{}.pkl'.format(name), 'rb'))
    labels = estimator.labels_
    index_list = [[] for _ in range(8)]
    for i, v in enumerate(labels):
        index_list[v].append(i)
    for i in range(len(index_list)):
        index_list[i] = random.sample(index_list[i], num_samples)
    return index_list


def clustered_yelp2file():
    """
    将yelp文本根据聚类列表(每类10000条样本)写入文件
    :return:
    """
    for positive in [True, False]:
        root = './dataset/yelp_clustered_csv/pos' if positive else './dataset/yelp_clustered_csv/neg'
        label = 1 if positive else 0

        yelp_text_list = _get_yelp_text_list(positive)

        index_list = _get_index_list(positive)
        for ii, _list in enumerate(index_list):
            csv_file = open(os.path.join(root, '{}.csv'.format(str(ii))), 'w')
            writer = csv.writer(csv_file)
            rows = [(label, yelp_text_list[i]) for i in _list]
            writer.writerows(rows)


def _tsne(positive, num_sample=500):
    index_list = np.array(_get_index_list(positive)).flatten()
    emb_pca = _pca(positive)[index_list]

    name = 'pos' if positive else 'neg'
    estimator = pickle.load(open('./count/estimator_{}.pkl'.format(name), 'rb'))
    labels = estimator.labels_[index_list]

    # 从样本中随机取500个出来做tsne
    random.seed(3)
    idx_list = random.sample(range(len(emb_pca)), num_sample)
    emb_pca = emb_pca[idx_list]
    labels = labels[idx_list]

    emb_tsne = TSNE(n_jobs=10, random_state=2).fit_transform(emb_pca)
    return emb_tsne, labels


def _ni_index(class_0, class_1):
    mean_0 = np.mean(class_0, axis=0)
    mean_1 = np.mean(class_1, axis=0)
    std = np.std(np.concatenate((class_0, class_1), axis=0))
    z = (mean_0 - mean_1) / std
    ni = np.linalg.norm(z)
    return ni


def get_ni_list(positive):
    """
    pos rank: [3, 4, 7, 0, 5, 2, 6, 1]
    neg rank: [0, 5, 3, 4, 2, 6, 1, 7]
    :param positive:
    :return:
    """
    # 获取列表并
    index_list = np.array(_get_index_list(positive))

    emb_pca = _pca(positive)[index_list]

    name = 'pos' if positive else 'neg'
    estimator = pickle.load(open('./count/estimator_{}.pkl'.format(name), 'rb'))
    labels = estimator.labels_[index_list]

    # 重新计算采样后的聚类中心,并取聚类中心模长最大的类作为基类
    cluster_centers = np.mean(emb_pca, axis=1)
    num_basic_class = np.argmax(np.linalg.norm(cluster_centers, axis=1))
    basic_class = emb_pca[num_basic_class]

    ni_list = []
    for i in range(len(emb_pca)):
        compared_class = emb_pca[i]
        ni_list.append(_ni_index(basic_class, compared_class))

    ni_rank = sorted(range(len(ni_list)), key=lambda k: ni_list[k])

    return ni_list, ni_rank


def clustered_yelp_file_combine():
    root = './dataset/yelp_combined_csv'

    _, pos_rank = get_ni_list(True)
    _, neg_rank = get_ni_list(False)

    for i, (p_rank, n_rank) in enumerate(zip(pos_rank, neg_rank)):
        dir = os.path.join(root, str(i))
        if not os.path.exists(dir):
            os.mkdir(os.path.join(root, str(i)))
        shutil.copy(f'./dataset/yelp_clustered_csv/pos/{p_rank}.csv', f'{root}/{str(i)}/pos.csv')
        shutil.copy(f'./dataset/yelp_clustered_csv/neg/{n_rank}.csv', f'{root}/{str(i)}/neg.csv')


def print_ni_info():
    pos_ni, pos_rank = get_ni_list(True)
    neg_ni, neg_rank = get_ni_list(False)
    print('POS')
    print(pos_ni)
    print(pos_rank)
    print('NEG')
    print(neg_ni)
    print(neg_rank)
    print('NI_SUM')
    ni_avg = [(pos_ni[rank1] + neg_ni[rank2]) / 2 for rank1, rank2 in zip(pos_rank, neg_rank)]
    print(ni_avg)


def plot():
    plt.figure(1)
    ax = plt.subplot(1, 3, 1)
    ax.set_title('Positive Samples')
    emb_tsne_pos, labels_pos = _tsne(True)
    # # 更换label标签
    labels_pos_ = []
    _, rank_pos = get_ni_list(True)
    for label in labels_pos:
        labels_pos_.append(rank_pos.index(label))
    plt.scatter(emb_tsne_pos[:, 0], emb_tsne_pos[:, 1], c=labels_pos_, cmap='Purples')

    ax = plt.subplot(1, 3, 2)
    ax.set_title('Negative Samples')
    emb_tsne_neg, labels_neg = _tsne(False)
    labels_neg_ = []
    _, rank_neg = get_ni_list(False)
    for label in labels_neg:
        labels_neg_.append(rank_neg.index(label))
    ax.scatter(emb_tsne_neg[:, 0], emb_tsne_neg[:, 1], c=labels_neg_, cmap='Blues')

    ax = plt.subplot(1, 3, 3)
    ax.set_title('Total Samples')
    ax.scatter(emb_tsne_pos[:, 0], emb_tsne_pos[:, 1], c=labels_pos_, cmap='Purples')
    ax.scatter(emb_tsne_neg[:, 0], emb_tsne_neg[:, 1], c=labels_neg_, cmap='Blues')

    plt.show()


clustered_yelp_file_combine()
