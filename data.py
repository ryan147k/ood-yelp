#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/03/27 Sat
# TIME: 20:50:10
# DESCRIPTION: Dataset类定义
import torch
from torch.utils.data import Dataset
import torchtext
from torchtext import vocab
from nltk import word_tokenize
from transformers import BertTokenizer, XLNetTokenizer
from allennlp.modules import elmo
import os
import linecache
import csv
import pickle


class YelpVocab(vocab.Vocab):
    def __init__(self, vectors, min_freq=1):
        dataset = torchtext.datasets.YelpReviewPolarity(root='./dataset', split='train')
        tokens = self._get_tokens(dataset)
        _vocab = vocab.build_vocab_from_iterator(tokens)
        super(YelpVocab, self).__init__(_vocab.freqs, min_freq=min_freq, vectors=vectors)
        
    def _get_tokens(self, dataset):
        for _, text in dataset:
            yield self._tokenizer(text.lower())

    def _tokenizer(self, word):
        return word_tokenize(word)


class _YelpClusterBaseDataset(Dataset):
    def __init__(self, _class: int, root='./dataset/yelp_combined_csv'):
        """
        _class: 数据类别
        """
        assert 0 <= _class < 8

        super(_YelpClusterBaseDataset, self).__init__()

        self.max_length = 128

        self.pos_data_path = os.path.join(root, f'{str(_class)}/pos.csv')
        self.neg_data_path = os.path.join(root, f'{str(_class)}/neg.csv')
        self.size = 20000

    def __getitem__(self, index):
        positive = True if index >= self.size / 2 else False
        data_path = self.pos_data_path if positive else self.neg_data_path
        index = int(index % (self.size / 2)) + 1
        line = linecache.getline(data_path, index)
        label, text = list(csv.reader([line]))[0]

        # 截取文本的前128个单词
        if len(text) > self.max_length:
            text = text[:self.max_length]

        return text, int(label)

    def __len__(self):
        return self.size


class YelpDatasetForBow(Dataset):
    def __init__(self, _class):
        super(YelpDatasetForBow, self).__init__()
        self.data = _YelpClusterBaseDataset(_class)
        self.vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))

    def __getitem__(self, index):
        text, label = self.data[index]
        text = torch.FloatTensor(self._token2bow(self._tokenizer(text)))
        label = torch.LongTensor([label])
        return text, label

    def __len__(self):
        return len(self.data)

    def _token2bow(self, tokens):
        bow = [0] * len(self.vocab)
        for token in tokens:
            idx = self.vocab.stoi[token]
            bow[idx] += 1
        return bow

    @staticmethod
    def _tokenizer(text):
        return word_tokenize(text)


class YelpDatasetForGlove(Dataset):
    def __init__(self, _class):
        super(YelpDatasetForGlove, self).__init__()

        self.data = _YelpClusterBaseDataset(_class)
        self.vocab = pickle.load(open('./YelpVocab.pkl', 'rb'))
    
    def __getitem__(self, index):
        text, label = self.data[index]
        # 转Tensor
        text = torch.LongTensor(self._token2idx(self._tokenizer(text)))
        label = torch.LongTensor([label])
        return text, label

    def __len__(self):
        return len(self.data)

    def _token2idx(self, tokens):
        return [self.vocab.stoi[token] for token in tokens]

    @staticmethod
    def _tokenizer(text):
        return word_tokenize(text)
    
    def collate_fn(self, batch):
        def _pad(seq, pad_idx):
            max_len = max([len(t) for t in seq])
            res = []
            for t in seq:
                pad = [pad_idx] * (max_len - len(t))
                t = torch.hstack((t, torch.LongTensor(pad)))
                res.append(t)
            res_tensor = torch.stack(res)
            return res_tensor

        text, label = zip(*batch)
        pad_idx = self.vocab.stoi['<pad>']
        text_pad = _pad(text, pad_idx)
        label = torch.hstack(label)
        return text_pad, label


class YelpDatasetForBert(Dataset):
    def __init__(self, _class):
        super(YelpDatasetForBert, self).__init__()

        self.data = _YelpClusterBaseDataset(_class)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        text, label = self.data[index]
        
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids = input.input_ids.squeeze()
        token_type_ids = input.token_type_ids.squeeze()
        attention_mask = input.attention_mask.squeeze()

        label = torch.LongTensor([label])

        return (input_ids, token_type_ids, attention_mask), label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        def _pad(seq, pad_idx):
            max_len = max([len(t) for t in seq])
            res = []
            for t in seq:
                pad = [pad_idx] * (max_len - len(t))
                t = torch.hstack((t, torch.LongTensor(pad)))
                res.append(t)
            res_tensor = torch.stack(res)
            return res_tensor

        data, label = zip(*batch)
        input_ids, token_type_ids, attention_mask = zip(*data)
        pad_idx = self.tokenizer.vocab['[PAD]']

        input_ids_pad = _pad(input_ids, pad_idx).unsqueeze(dim=0)
        token_type_ids_pad = _pad(token_type_ids, pad_idx).unsqueeze(dim=0)
        attention_mask_pad = _pad(attention_mask, pad_idx).unsqueeze(dim=0)

        data = torch.cat((input_ids_pad, token_type_ids_pad, attention_mask_pad), dim=0)
        label = torch.hstack(label)

        return data, label


class YelpDatasetForXLNet(Dataset):
    def __init__(self, _class):
        super(YelpDatasetForXLNet, self).__init__()

        self.data = _YelpClusterBaseDataset(_class=_class)
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

    def __getitem__(self, index):
        text, label = self.data[index]

        input = self.tokenizer(text, padding=True, max_length=512, truncation=True, return_tensors='pt')
        input_ids = input.input_ids.squeeze()
        token_type_ids = input.token_type_ids.squeeze()
        attention_mask = input.attention_mask.squeeze()

        label = torch.LongTensor([label])

        return (input_ids, token_type_ids, attention_mask), label

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        def _pad(seq, pad_idx):
            max_len = max([len(t) for t in seq])
            res = []
            for t in seq:
                pad = [pad_idx] * (max_len - len(t))
                t = torch.hstack((t, torch.LongTensor(pad)))
                res.append(t)
            res_tensor = torch.stack(res)
            return res_tensor

        data, label = zip(*batch)
        input_ids, token_type_ids, attention_mask = zip(*data)
        pad_idx = self.tokenizer.pad_token_id

        input_ids_pad = _pad(input_ids, pad_idx).unsqueeze(dim=0)
        token_type_ids_pad = _pad(token_type_ids, pad_idx).unsqueeze(dim=0)
        attention_mask_pad = _pad(attention_mask, pad_idx).unsqueeze(dim=0)

        data = torch.cat((input_ids_pad, token_type_ids_pad, attention_mask_pad), dim=0)
        label = torch.hstack(label)

        return data, label


class YelpDatasetForElmo(Dataset):
    def __init__(self, _class):
        super(YelpDatasetForElmo, self).__init__()

        self.data = _YelpClusterBaseDataset(_class=_class)

    def __getitem__(self, index):
        text, label = self.data[index]

        text = self._token2idx(self._tokenizer(text)).squeeze(dim=0)
        text = self._truncate(text)
        label = torch.LongTensor([label])
        return text, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _token2idx(tokens):
        return elmo.batch_to_ids([tokens])

    @staticmethod
    def _tokenizer(text):
        return word_tokenize(text)

    @staticmethod
    def _truncate(text):
        MAXLENGTH = 512
        if len(text) > MAXLENGTH:
            text = text[:MAXLENGTH]
        return text

    def collate_fn(self, batch):
        def _pad(seq, pad_idx):
            max_len = max([len(t) for t in seq])
            res = []
            for t in seq:
                pad = [[pad_idx] * 50 for i in range(max_len - len(t))]
                t = torch.cat((t, torch.LongTensor(pad)), dim=0)
                res.append(t)
            res_tensor = torch.stack(res)
            return res_tensor

        text, label = zip(*batch)
        pad_idx = 0
        text_pad = _pad(text, pad_idx)
        label = torch.hstack(label)
        return text_pad, label


if __name__ == "__main__":
    dataset = YelpDatasetForBow(0)
    a = dataset[0]
    pass