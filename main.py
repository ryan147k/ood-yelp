#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Rran Hu
# DATE: 2021/03/30 Tue
# TIME: 13:08:06
# DESCRIPTION: 模型训练和测试
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from model import Bow, WordAvg, DPCNN, BiLSTM, BertBase, ELMo, XLNetBase, SelfAttention
from data import YelpVocab, YelpDatasetForGlove, YelpDatasetForBert, YelpDatasetForElmo, YelpDatasetForXLNet
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import os
import random
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--ex_num', type=str)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--print_iter', type=int, default=20)
parser.add_argument('--dropout', type=float, default=0.5)
args = parser.parse_args()


def train(model,
          save_path: str,
          ex_name: str,
          train_dataset,
          val_dataset,
          test_datasets=None):
    # data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=3, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                            num_workers=3, collate_fn=val_dataset.collate_fn)
    if test_datasets is None:
        test_loaders = []
    else:
        test_loaders = [DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                                   num_workers=2, collate_fn=dataset.collate_fn)
                        for dataset in test_datasets]

    # model

    # 多GPU运行
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # train

    writer = SummaryWriter()
    best_val_acc, best_val_iter = 0.0, None  # 记录全局最优信息
    save_model = False

    iter = 0
    for epoch in range(args.epoch_num):

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # forward
            y_hat = model(batch_x)
            loss = loss_fn(y_hat, batch_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()

            # 计算精度
            _, pred = y_hat.max(1)
            num_correct = (pred == batch_y).sum().item()
            acc = num_correct / len(batch_y)

            iter += 1
            if iter % args.print_iter == 0:
                # 打印信息
                train_loss, train_acc = loss.item(), acc
                val_loss, val_acc = val(model, val_loader)
                test_info_list = [val(model, loader) for loader in test_loaders]
                print("\n[INFO] Epoch {} Iter {}:\n \
                            \tTrain: Loss {:.4f}, Accuracy {:.4f}\n \
                            \tVal:   Loss {:.4f}, Accuracy {:.4f}".format(epoch + 1, iter,
                                                                          train_loss, train_acc,
                                                                          val_loss, val_acc))
                for ii, (test_loss, test_acc) in enumerate(test_info_list):
                    print("\tTest{}: Loss {:.4f}, Accuracy {:.4f}".format(ii, test_loss, test_acc))

                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(save_path.split('/')[-1], 'acc'),
                                  value_dict={'train_acc': train_acc,
                                              'val_acc': val_acc},
                                  x_axis=iter)
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(save_path.split('/')[-1], 'loss'),
                                  value_dict={'train_loss': train_loss,
                                              'val_loss': val_loss},
                                  x_axis=iter)

                # 更新全局最优信息
                if val_acc > best_val_acc:
                    best_val_acc, best_val_iter = val_acc, iter
                    save_model = True
                if save_model:
                    # 保存模型
                    torch.save(model.module.state_dict(), '{}_best.pt'.format(save_path))
                    save_model = False

                print("\t best val   acc so far: {:.4} Iter: {}".format(best_val_acc, best_val_iter))

                torch.save(model.module.state_dict(), '{}.pt'.format(save_path, iter))

        # 保存模型
        # torch.save(model.module.state_dict(), '{}_e{}.pt'.format(save_path, epoch))


@torch.no_grad()
def val(model, dataloader):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0
    acc_sum = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # forward
        y_hat = model(batch_x)
        loss = loss_fn(y_hat, batch_y)

        loss_sum += loss.item()
        # 计算精度
        _, pred = y_hat.max(1)
        num_correct = (pred == batch_y).sum().item()
        acc = num_correct / len(batch_y)
        acc_sum += acc

    model.train()

    return loss_sum / len(dataloader), acc_sum / len(dataloader)


def tensorboard_write(writer, ex_name: str, mode_name, value_dict, x_axis):
    """
    tensorboardX 作图
    :param writer: writer = SummaryWriter()
    :param ex_name: 实验名称
    :param mode_name: 模型名称+数据 eg. resnet18 acc
    :param value_dict:
    :param x_axis:
    :return:
    """
    writer.add_scalars(main_tag='{}/{}'.format(ex_name, mode_name),
                       tag_scalar_dict=value_dict,
                       global_step=x_axis)


class Experiment:
    """
    记录每一次的实验设置
    """
    # wordavg = WordAvg()
    # bilstm = BiLSTM(args.dropout)
    # dpcnn = DPCNN()
    # bert = BertBase()
    # elmo = ELMo(args.dropout)
    # xlnet = XLNetBase()
    # selfatt = SelfAttention(args.dropout)

    @staticmethod
    def _mkdir(save_dir):
        """
        如果目录不存在, 则创建
        :param save_dir: 模型检查点保存目录
        :return:
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @staticmethod
    def _get_loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    @staticmethod
    def _dataset_split(dataset):
        """
        将dataset分成两个dataset: train & val
        :param dataset:
        :return:
        """
        class _Dataset(Dataset):
            def __init__(self, _train=True):
                self.dataset = dataset

                index_list = list(range(len(dataset)))
                random.seed(2)
                random.shuffle(index_list)

                split_num = int(len(dataset) * 9 / 10)
                train_index_list = index_list[:split_num]
                val_index_list = index_list[split_num:]

                self.index_list = train_index_list if _train else val_index_list

            def __getitem__(self, index):
                data, label = self.dataset[self.index_list[index]]
                return data, label

            def __len__(self):
                return len(self.index_list)

            def collate_fn(self, batch):
                return self.dataset.collate_fn(batch)

        return _Dataset(_train=True), _Dataset(_train=False)

    @classmethod
    def _basic_test(cls, model, dataset):
        """
        获取模型在某个测试集上的loss和acc
        :param model:
        :param dataset:
        :return:
        """
        model = nn.parallel.DataParallel(model)
        model.to(device)

        loader = cls._get_loader(dataset)
        loss, acc = val(model, loader)
        # print('loss {} acc {}'.format(loss, acc))
        return loss, acc

    @classmethod
    def _test(cls, model, save_dir, model_name, datasets, ckpt_nums):
        """
        获取一系列模型检查点在测试集上的准确率
        :param model: 待测试模型
        :param save_dir: 模型检查点保存目录
        :param model_name: 模型检查点名称
        :param datasets: 测试集列表
        :param ckpt_nums: 检查点iter (同时是x轴坐标)
        :return:
        """
        acc_tests = [[] for _ in datasets]
        # 记录每个epoch的模型在数据集上的准确率
        for num in tqdm(ckpt_nums):
            model.load_state_dict(torch.load(f'{save_dir}/{model_name}_e{num}.pt'))
            # print('[INFO] Iter {}'.format(num), end='\n\t')
            for i, dataset in enumerate(datasets):
                _, acc = cls._basic_test(model, dataset)
                acc_tests[i].append(acc)
        return acc_tests

    @staticmethod
    def _plot(scalars_list, labels, x_axis):
        """
        画曲线图
        :param scalars_list: List[List], 每一个子List就是一条曲线
        :param labels: 每一个子List所代表的标签
        :param x_axis: x轴数值
        :return:
        """
        for i in range(len(scalars_list)):
            plt.plot(x_axis, scalars_list[i], label=labels[i])
            plt.xlabel('Iter')
            plt.ylabel('Acc')
            plt.legend()
            plt.show()

    @classmethod
    def ex1(cls):
        args.batch_size = 512
        args.epoch_num = 100
        print(args)

        ex_name = 'ex1'
        save_dir = './ckpts/ex1/0923'
        cls._mkdir(save_dir)

        model = WordAvg()
        train_dataset, val_dataset = cls._dataset_split(YelpDatasetForGlove(0))
        train(model, f'{save_dir}/{model.__class__.__name__}', ex_name, train_dataset, val_dataset)

    @classmethod
    def ex2(cls):
        args.batch_size = 512
        args.epoch_num = 100
        print(args)

        ex_name = 'ex2'
        save_dir = './ckpts/ex2/0923'
        cls._mkdir(save_dir)

        model = BiLSTM()
        train_dataset, val_dataset = cls._dataset_split(YelpDatasetForGlove(0))
        train(model, f'{save_dir}/{model.__class__.__name__}', ex_name, train_dataset, val_dataset)

    @classmethod
    def ex3(cls):
        args.batch_size = 512
        args.epoch_num = 100
        print(args)

        ex_name = 'ex3'
        save_dir = './ckpts/ex3/0923'
        cls._mkdir(save_dir)

        model = DPCNN()
        train_dataset, val_dataset = cls._dataset_split(YelpDatasetForGlove(0))
        train(model, f'{save_dir}/{model.__class__.__name__}', ex_name, train_dataset, val_dataset)

    @classmethod
    def ex4(cls):
        args.batch_size = 128
        args.epoch_num = 100
        print(args)

        ex_name = 'ex4'
        save_dir = './ckpts/ex4/0923'
        cls._mkdir(save_dir)

        model = SelfAttention()
        train_dataset, val_dataset = cls._dataset_split(YelpDatasetForGlove(0))
        train(model, f'{save_dir}/{model.__class__.__name__}', ex_name, train_dataset, val_dataset)

    @classmethod
    def ex5(cls):
        """测试用"""
        args.batch_size = 8
        args.epoch_num = 100
        print(args)

        # ex_name = 'ex4'
        # save_dir = './ckpts/ex4/0923'
        # cls._mkdir(save_dir)

        model = BertBase()
        train_dataset, val_dataset = cls._dataset_split(YelpDatasetForBert(0))
        # train(model, f'{save_dir}/{model.__class__.__name__}', ex_name, train_dataset, val_dataset)
        print(cls._basic_test(model, val_dataset))


args.ex_num = '5'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ex = getattr(Experiment, f'ex{args.ex_num.strip()}')
ex()
