#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Used to visualize data distribution

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import wandb
from statistics import pstdev

from utils.sampling import mnist_iid, mnist_non_iid, cifar_iid, cifar_non_iid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Fed import FedLearn
from models.Fed import model_deviation
from models.test import test_img

import tables
import yaml
import glob
import json
import re

from PIL import Image

from pysnn.datasets import nmnist_train_test

if __name__ == '__main__':
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_keys = None
    h5fs = None
    # load dataset and split users
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    dict_users = cifar_non_iid(dataset_train, num_classes=10, num_users=100, alpha=0.2, min_datasize=32)

    # user_ids = [i for i in range(100)]
    # class_tally = [[0 for user_id in range(100)] for c in range(10)]
    # for user_id in user_ids:
    #     user_train_dataset = DatasetSplit(dataset_train, dict_users[user_id])
    #     class_count = [0 for i in range(10)]
    #     for i in range(len(user_train_dataset)):
    #         image, label = user_train_dataset[i]
    #         class_count[label] += 1
    #     print("User {}, class_count: {}".format(user_id, class_count))
    #     for c in range(10):
    #         class_tally[c][user_id] += class_count[c]
    
    # for c in range(10):
    #     s = np.array(class_tally[c])
    #     user_rank = list(np.argsort(s))[::-1]
    #     print("Class {}, user ranking {}".format(c, user_rank[:10]))

    # compute total training data distribution (by classes) of each run
    run_names = [
        "CF10_100c100c10_0.2_bl_rd",
    ]
    for run_name in run_names:
        chosen_users_list = []
        file_name = "./experiments/" + run_name + "/client_selection_history.txt"
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                res = re.findall(r'\[.*?\]', line)
                if res != []:
                    l = res[0][1:-1].split(", ")
                    for i in range(len(l)):
                        l[i] = int(l[i])
                    chosen_users_list.append(l)
        # assert(len(chosen_users_list) == 60)

        means, stds = [], []
        for chosen_users in chosen_users_list[:60]:
            class_tally = [0 for c in range(args.num_classes)]
            for user_id in chosen_users:
                user_train_dataset = DatasetSplit(dataset_train, dict_users[user_id])
                for i in range(len(user_train_dataset)):
                    image, label = user_train_dataset[i]
                    class_tally[label] += 1
            print("Total training data by class: {}, mean {}, std {}".format(class_tally, sum(class_tally)/len(class_tally), pstdev(class_tally)))
            means.append(sum(class_tally)/len(class_tally))
            stds.append(pstdev(class_tally))
        
        # overall mean
        print("{}: mean of mean {}, mean of std {}".format(run_name, sum(means)/len(means), sum(stds)/len(stds)))


