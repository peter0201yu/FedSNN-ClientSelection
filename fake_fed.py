#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Perform fake fed learning: pass around model, no aggregation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch
import torch.nn as nn
import wandb

from utils.sampling import mnist_iid, mnist_non_iid, cifar_iid, cifar_non_iid
from utils.options import args_parser
from models.Update import LocalUpdate, DatasetSplit
from models.Fed import FedLearn
from models.Fed import model_deviation
from models.test import test_img
import models.vgg_spiking_bntt as snn_models_bntt
from models.simple_conv_cf10 import Simple_CF10_BNTT
from models.simple_conv_mnist import Simple_Mnist_BNTT, Simple_Mnist_NoBNTT, Simple_Mnist_BNTT_Rate
import models.client_selection as client_selection
import models.candidate_selection as candidate_selection

import tables
import yaml
import glob
import json

from PIL import Image

from pysnn.datasets import nmnist_train_test

if __name__ == '__main__':
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if args.wandb:
        wandb.init(project=args.project, name=args.wandb,
                    config={"epochs": args.epochs,  "dataset": args.dataset, "alpha": args.alpha},
                    notes="train with data of a random user, as if only aggregating the update from that one user")

    dataset_keys = None
    h5fs = None
    # load dataset and split users
    if args.dataset == 'CIFAR10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users, args.alpha)
    elif args.dataset == 'CIFAR100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users, args.alpha)
    elif args.dataset == 'EMNIST':
        # same transform and splitting as MNIST
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.EMNIST('../data/emnist', 'bymerge', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.EMNIST('../data/emnist', 'bymerge', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_non_iid(dataset_train, args.num_classes, args.num_users, args.alpha)
    else:
        exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # build model
    model_args = {'args': args}
    if args.model[0:3].lower() == 'vgg':
        if args.snn:
            model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps}
            net = snn_models_bntt.SNN_VGG9_BNTT(**model_args).cuda()
    elif args.model == 'simple':
        model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps, 'img_size': args.img_size}
        if args.dataset == 'EMNIST':
            if args.bntt:
                if args.direct:
                    net = Simple_Mnist_BNTT(**model_args).cuda()
                else:
                    net = Simple_Mnist_BNTT_Rate(**model_args).cuda()
            else:
                model_args['leak_mem'] = 0.5
                net = Simple_Mnist_NoBNTT(**model_args).cuda()
        else:
            if args.bntt:
                net = Simple_CF10_BNTT(**model_args).cuda()
            else:
                model_args['leak_mem'] = 0.5
                net = VGG5_CF10_NoBNTT(**model_args).cuda()
    else:
        exit('Error: unrecognized model')
    print(net)
    net = torch.nn.DataParallel(net)

    # training
    loss_train_list = []

    # metrics to store
    ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []

    # print("len(dict_users[0]): ", len(dict_users[0]))

    loss_func = nn.CrossEntropyLoss()

    # train and update
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay = 1e-4)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr = args.lr)
    else:
        print("Invalid optimizer")

    for epoch in range(args.epochs):
        user_id = np.random.choice(args.num_users)
    
        net.train()
        ldr_train = DataLoader(DatasetSplit(dataset=dataset_train, idxs=dict_users[user_id]), batch_size=args.local_bs, shuffle=True, drop_last=True)
        print("ROUND {}, training w/ {} images of user {}".format(epoch, len(ldr_train.dataset), user_id))

        epoch_loss = []
        trained_data_size = 0
        for local_ep in range(args.local_ep):
            print("local epoch: ", local_ep)
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(args.device), labels.to(args.device)
                trained_data_size += len(images)
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        local_ep, batch_idx * len(images), num_samples,
                               100. * batch_idx / len(ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # avg of all local epochs (In federated learning, only keeps track of FedLearn epochs)
        loss_train_list.append(sum(epoch_loss) / len(epoch_loss))
    
        if epoch % args.eval_every == 0:
            # testing
            net.eval()
            acc_train, loss_train = test_img(net, dataset_train, args)
            print("Round {:d}, Training accuracy: {:.2f}".format(epoch, acc_train))
            acc_test, loss_test = test_img(net, dataset_test, args)
            print("Round {:d}, Testing accuracy: {:.2f}".format(epoch, acc_test))
    
            # Add metrics to store
            ms_acc_train_list.append(acc_train)
            ms_acc_test_list.append(acc_test)
            ms_loss_train_list.append(loss_train)
            ms_loss_test_list.append(loss_test)

            if args.wandb:
                wandb.log({"server_train_loss": loss_train, "server_test_loss": loss_test, 
                            "server_train_acc": acc_train, "server_test_acc": acc_test, "Round": epoch+1})

    Path('./{}'.format(args.result_dir)).mkdir(parents=True, exist_ok=True)
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train_list)), loss_train_list)
    plt.ylabel('train_loss')
    plt.savefig('./{}/fed_loss_{}_{}_{}_C{}_iid{}.png'.format(args.result_dir,args.dataset, args.model, args.epochs, args.frac, args.iid))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(ms_acc_train_list)), ms_acc_train_list)
    plt.plot(range(len(ms_acc_test_list)), ms_acc_test_list)
    plt.plot()
    plt.yticks(np.arange(0, 100, 10))
    plt.ylabel('Accuracy')
    plt.legend(['Training acc', 'Testing acc'])
    plt.savefig('./{}/fed_acc_{}_{}_{}_C{}_iid{}.png'.format(args.result_dir, args.dataset, args.model, args.epochs, args.frac, args.iid))

    # Write metric store into a CSV
    metrics_df = pd.DataFrame(
        {
            'Train acc': ms_acc_train_list,
            'Test acc': ms_acc_test_list,
            'Train loss': ms_loss_train_list,
            'Test loss': ms_loss_test_list
        })
    metrics_df.to_csv('./{}/fed_stats_{}_{}_{}_C{}_iid{}.csv'.format(args.result_dir, args.dataset, args.model, args.epochs, args.frac, args.iid), sep='\t')