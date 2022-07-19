#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.nn.functional as F
import numpy as np
import random
from sklearn import metrics
import sys
import os
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = DatasetSplit(dataset, idxs)
        if self.args.train_frac is not None:
            num_samples = int(len(self.dataset) * self.args.train_frac)
            sampler = RandomSampler(self.dataset, num_samples=num_samples)
            self.ldr_train = DataLoader(self.dataset, batch_size=self.args.local_bs, sampler=sampler, drop_last=True)
        else:
            self.ldr_train = DataLoader(self.dataset, batch_size=self.args.local_bs, shuffle=True, drop_last=True)

    def train(self, net, local_epochs=None):
        net.train()
        # train and update
        if self.args.optimizer == "SGD":
            if self.args.bntt:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
            else:
                optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.8, weight_decay=0)
        elif self.args.optimizer == "Adam":
            optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr)
        else:
            print("Invalid optimizer")

        epoch_loss = []
        trained_data_size = 0
        if local_epochs is None:
            local_epochs = self.args.local_ep
        # activities = []
        for iter in range(local_epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                trained_data_size += len(images)
                net.zero_grad()
                log_probs = net(images)
                # activities.append(activity)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # print("training sample size: ", trained_data_size)
        # activity = torch.mean(torch.stack(activities), 0)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), trained_data_size
    
    def test_with_train_data(self, net):
        net.eval()
        # testing
        test_loss = 0
        correct = 0
        if self.args.test_size:
            test_size = min(len(self.dataset), self.args.test_size)
            sampler = RandomSampler(self.dataset, num_samples=test_size)
            data_loader = DataLoader(self.dataset, sampler=sampler, batch_size=self.args.bs, drop_last=True)
        else:
            data_loader = DataLoader(self.dataset, batch_size=self.args.bs, drop_last=True)
            test_size = len(data_loader.dataset)
        
        print("Testing on {} images".format(test_size))
        # l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.cuda(), target.cuda()
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= test_size
        accuracy = 100.00 * correct / test_size
        if self.args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, test_size, accuracy))
        return accuracy.item(), test_loss
        

