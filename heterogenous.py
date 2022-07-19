#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Cleaned up version of main_fed.py

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
import models.vgg_spiking_bntt as snn_models_bntt
# import models.vgg as ann_models
# from models.simple_conv_cf10 import Simple_CF10_BNTT
# from models.simple_conv_mnist import Simple_Mnist_BNTT, Simple_Mnist_NoBNTT, Simple_Mnist_BNTT_Rate

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
                    config={"epochs": args.epochs, "num_users": args.num_users, "frac_users": args.frac, "dataset": args.dataset, "alpha": args.alpha, "timestep_mean": args.timestep_mean, "timestep_std": args.timestep_std, "timestep_pattern": args.timestep_pattern})

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

    # To build global model, need max number of timesteps
    # timesteps_list = np.random.normal(loc=args.timestep_mean, scale=args.timestep_std, size=args.num_users)
    # max_timestep = int(args.timestep_mean + args.timestep_std * 3)
    max_timestep = 60
    print("Build global model with max timestep: ", max_timestep)

    model_args = {'args': args}
    if args.model[0:3].lower() == 'vgg':
        if args.snn:
            model_args = {'num_cls': args.num_classes, 'timesteps': args.timestep_mean, 'max_timestep': max_timestep, 'img_size': args.img_size}
            net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).cuda()
        else:
            model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': 3, 'dropout': args.dropout}
            net_glob = ann_models.VGG(**model_args).cuda()
    elif args.model == 'simple':
        model_args = {'num_cls': args.num_classes, 'timesteps': args.timestep_mean, 'max_timestep': max_timestep, 'img_size': args.img_size}
        if args.dataset == 'EMNIST':
            if args.bntt:
                if args.direct:
                    net_glob = Simple_Mnist_BNTT(**model_args).cuda()
                else:
                    net_glob = Simple_Mnist_BNTT_Rate(**model_args).cuda()
            else:
                model_args['leak_mem'] = 0.5
                net_glob = Simple_Mnist_NoBNTT(**model_args).cuda()
        else:
            if args.bntt:
                net_glob = Simple_CF10_BNTT(**model_args).cuda()
            else:
                model_args['leak_mem'] = 0.5
                net_glob = VGG5_CF10_NoBNTT(**model_args).cuda()
    else:
        exit('Error: unrecognized model')
    # print(net_glob)

    # copy weights
    if args.pretrained_model:
        net_glob.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))

    net_glob = nn.DataParallel(net_glob)
    # training
    loss_train_list = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # metrics to store
    ms_acc_train_list, ms_loss_train_list = [], []
    ms_acc_test_list, ms_loss_test_list = [], []
    # ms_model_deviation = []

    # testing
    net_glob.eval()
    acc_train, loss_train = 0, 0
    acc_test, loss_test = 0, 0
    # Add metrics to store
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

    # Define LR Schedule
    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))
    print("lr_interval: ", lr_interval)

    # Define Fed Learn object
    fl = FedLearn(args)

    # federated learning constants 
    m = max(int(args.frac * args.num_users), 1)
    num_candidates = m*2

    chosen_users = None
    client_set = set()

    for iter in range(args.epochs):
        print("--------------------------------------------------")
        print("Round {}, Learning rate {}".format(iter+1, args.lr))
        w_locals_all, loss_locals_all, trained_data_size_all = [], [], []

        # Get a new timestep distribution
        timesteps_list = np.random.normal(loc=args.timestep_mean, scale=args.timestep_std, size=args.num_users)

        candidates = np.random.choice(range(args.num_users), num_candidates, replace=False)
        # print("Selected candidates randomly: ", candidates)

        if args.client_selection == "biggest_loss":
            net_glob.eval()
            tmp_losses = []
            for idx in candidates:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
                model_args = {'num_cls': args.num_classes, 'timesteps': max(1, round(timesteps_list[idx])), 'max_timestep': max_timestep}
                model_copy = type(net_glob.module)(**model_args) # get a new instance
                model_copy = nn.DataParallel(model_copy)
                model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff
                tmp_acc, tmp_loss = local.test_with_train_data(net=model_copy.to(args.device))
                tmp_losses.append(tmp_loss)
            ret = sorted(list(range(len(tmp_losses))), key=lambda x: tmp_losses[x], reverse=True)
            chosen_users = [candidates[idx] for idx in ret[:m]]
            print("Selected users by biggest pre-train loss: ", chosen_users)
        else:
            chosen_users = np.random.choice(candidates, m, replace=False)
            print("Selected users randomly: ", chosen_users)

        client_set |= set(chosen_users)
        if args.wandb:
            wandb.log({"diff_client_num":len(client_set), "Round": iter+1})

        for counter, idx in enumerate(chosen_users):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
            model_args = {'num_cls': args.num_classes, 'timesteps': max(1, round(timesteps_list[idx])), 'max_timestep': max_timestep}
            model_copy = type(net_glob.module)(**model_args) # get a new instance
            model_copy = nn.DataParallel(model_copy)
            model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff

            # tmp_acc, tmp_loss = local.test_with_train_data(net=model_copy.to(args.device))
            # print("Estimate loss: ", tmp_loss)

            w, loss, trained_data_size = local.train(net=model_copy.to(args.device))
            w_locals_all.append(copy.deepcopy(w))
            loss_locals_all.append(copy.deepcopy(loss))
            trained_data_size_all.append(trained_data_size)

            # compute update norm
            # delta_w = {}
            # w_init = net_glob.state_dict()
            # for k in w_init.keys():
            #     delta_w[k] = w[k] - w_init[k]
            # norm = 0
            # for k in delta_w.keys():
            #     norm += float(torch.linalg.norm(delta_w[k].float()))
            # print("Local (weighted) update norm: ", norm * agg_weights[counter])

        # model_dev_list = model_deviation(w_locals_all, net_glob.state_dict())
        # ms_model_deviation.append(model_dev_list)

        # update global weights
        chosen_timesteps = [max(1, round(timesteps_list[idx])) for idx in chosen_users]
        if args.FedAvgWeight == "timestep_prop":
            agg_weights = [timestep / sum(chosen_timesteps) * len(chosen_users) for timestep in chosen_timesteps]
        elif args.FedAvgWeight == "timestep_inv":
            inverse_chosen_timesteps = [1/(timestep) for timestep in chosen_timesteps]
            agg_weights = [val / sum(inverse_chosen_timesteps) * len(chosen_users) for val in inverse_chosen_timesteps]
        elif args.FedAvgWeight == "train_loss_prop":
            agg_weights = [loss / sum(loss_locals_all) * len(chosen_users) for loss in loss_locals_all]
        elif args.FedAvgWeight == "train_loss_inv":
            inverse_train_loss = [1/(loss) for loss in loss_locals_all]
            agg_weights = [val / sum(inverse_train_loss) * len(chosen_users) for val in inverse_train_loss]
        else:
            agg_weights = [1 for i in range(len(chosen_users))]
        print("Perform weighted FedAvg by {}, weights {}".format(args.FedAvgWeight, agg_weights))

        w_glob = fl.FedAvgWeighted(w_locals_all, agg_weights, w_init = net_glob.state_dict())
        
        # delta_w = {}
        # w_init = net_glob.state_dict()
        # for k in w_init.keys():
        #     delta_w[k] = w_glob[k] - w_init[k].cpu()
        # norm = 0
        # for k in delta_w.keys():
        #     norm += float(torch.linalg.norm(delta_w[k].float()))
        # print("Global update norm: ", norm)
    
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
 
        loss_avg = sum(loss_locals_all) / len(loss_locals_all)
        print('Round {:3d}, Average loss {:.3f}'.format(iter+1, loss_avg))
        loss_train_list.append(loss_avg)
        if args.wandb:
            wandb.log({"client_avg_loss":loss_avg, "Round": iter+1})

        if iter % args.eval_every == 0:
            # testing
            net_glob.eval()
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            print("Round {:d}, Training accuracy: {:.2f}".format(iter+1, acc_train))
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:d}, Testing accuracy: {:.2f}".format(iter+1, acc_test))

            if args.wandb:
                wandb.log({"server_train_loss": loss_train, "server_test_loss": loss_test, 
                            "server_train_acc": acc_train, "server_test_acc": acc_test, "Round": iter+1})

            # Add metrics to store
            ms_acc_train_list.append(acc_train)
            ms_acc_test_list.append(acc_test)
            ms_loss_train_list.append(loss_train)
            ms_loss_test_list.append(loss_test)

        if iter in lr_interval:
            args.lr = args.lr/args.lr_reduce

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

    # torch.save(net_glob.module.state_dict(), './{}/saved_model'.format(args.result_dir))

    # fn = './{}/model_deviation_{}_{}_{}_C{}_iid{}.json'.format(args.result_dir, args.dataset, args.model, args.epochs, args.frac, args.iid)
    # with open(fn, 'w') as f:
    #     json.dump(ms_model_deviation, f)
