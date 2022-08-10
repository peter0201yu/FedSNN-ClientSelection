#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# Cleaned up version of the original main_fed.py

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
import models.vgg as ann_models
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
                    config={"epochs": args.epochs, "num_users": args.num_users, "frac_users": args.frac, "client_selection": args.client_selection, "dataset": args.dataset, "alpha": args.alpha, "candidate_selection": args.candidate_selection, 
                    "candidate_frac": args.candidate_frac, "gamma": args.gamma})

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
            net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).cuda()
        else:
            model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': 3, 'dropout': args.dropout}
            net_glob = ann_models.VGG(**model_args).cuda()
    elif args.model == 'simple':
        model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps, 'img_size': args.img_size}
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
                net = Simple_CF10_BNTT(**model_args).cuda()
            else:
                model_args['leak_mem'] = 0.5
                net = VGG5_CF10_NoBNTT(**model_args).cuda()
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

    client_selection_history, client_set, dropped_clients = [], set(), []

    # federated learning constants 
    num_candidates = max(int(args.candidate_frac * args.num_users), 1)
    m = max(int(args.frac * args.num_users), 1)
    dataset_size = sum([len(dict_users[i]) for i in range(args.num_users)])
    data_probs = [float(len(dict_users[i]))/dataset_size for i in range(args.num_users)]
    chosen_candidates, chosen_users = None, None
    
    # based on data distribution: pick clients with most samples in all classes
    if args.client_selection == "handpick":
        handpick_list = [
            [87,27,98,72,62,37,17,61,36,33],
            [4, 29,96,39,32,89,70,15,69,2],
            [38,58,70,60,4, 19,49,94,65,92],
            [89,91,14,53,82,67,34,55,18,56],
            [35,40,93,88,26,85,64,68,21,45],
            [3, 7, 69,59,10,57,71,76,57,30],
            [67,45,84,86,73,55,22,64,3, 74],
            [86,83,47,7, 54,58,83,36,35,28],
            [82,35,39,95,9, 25,74,90,12,38],
            [73,31,72,8, 52,48,53,74,0, 71]
        ]


    for iter in range(args.epochs):
        print("Learning rate: ", args.lr)
        net_glob.train()
        # w_locals_selected, loss_locals_selected = [], []
        w_locals_all, loss_locals_all = [], []
        trained_data_size_all = []
        
        if args.candidate_selection == "random":
            candidates = candidate_selection.random(args.num_users, num_candidates)
        elif args.candidate_selection == "loop":
            candidates = candidate_selection.loop(args.num_users, num_candidates, iter)
        elif args.candidate_selection == "data_amount":
            candidates = candidate_selection.data_amount(args.num_users, num_candidates, data_probs)
        elif args.candidate_selection == "reduce_collision":
            candidates = candidate_selection.reduce_collision(args.num_users, num_candidates, data_probs, chosen_candidates, chosen_users, args.gamma)
        elif args.candidate_selection == "keep_good_avoid_bad":
            candidates = candidate_selection.keep_good_avoid_bad(args.num_users, num_candidates, data_probs, chosen_candidates, chosen_users, ms_acc_train_list, args.gamma, iter, dropped_clients)
        
        chosen_candidates = copy.deepcopy(candidates)
        print("candidate clients: ", candidates)
        
        if args.client_selection == "random":
            idxs_users = client_selection.random(len(candidates), m)
        elif "loss" in args.client_selection:
            tmp_losses = []
            for idx in candidates:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
                model_copy = type(net_glob.module)(**model_args) # get a new instance
                model_copy = nn.DataParallel(model_copy)
                model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff
                tmp_acc, tmp_loss = local.test_with_train_data(net=model_copy.to(args.device))
                tmp_losses.append(tmp_loss)
            if args.client_selection == "biggest_loss":
                idxs_users = client_selection.biggest_loss(tmp_losses, len(candidates), m)
            elif args.client_selection == "middle_loss":
                idxs_users = client_selection.middle_loss(tmp_losses, len(candidates), m)
            elif args.client_selection == "smallest_loss":
                idxs_users = client_selection.smallest_loss(tmp_losses, len(candidates), m)
            elif args.client_selection == "mixed_loss":
                idxs_users = client_selection.mixed_loss(tmp_losses, len(candidates), m)
            else:
                exit('Error: unrecognized client selection')
        elif args.client_selection == "grad_diversity" or args.client_selection == "spike_diversity":
            delta_w_locals_all = []
            w_init = net_glob.state_dict()
            activities = []
            for idx in candidates:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
                model_copy = type(net_glob.module)(**model_args) # get a new instance
                model_copy = nn.DataParallel(model_copy)
                model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff
                w, loss, trained_data_size, activity = local.train(net=model_copy.to(args.device), local_epochs=1)
                activities.append(activity)

            if args.client_selection == "grad_diversity":
                delta_w = {}
                for k in w_init.keys():
                    delta_w[k] = w[k] - w_init[k]
                delta_w_locals_all.append(delta_w)
                idxs_users = client_selection.grad_diversity(delta_w_locals_all, len(candidates), m)
            elif args.client_selection == "spike_diversity":
                idxs_users = client_selection.spike_diversity(activities, len(candidates), m)

        elif args.client_selection == "handpick":
            chosen_users = handpick_list[iter%len(handpick_list)]

        # idxs_users gives the client's index in the candidates list, need to convert
        if args.client_selection != "handpick":
            chosen_users = [candidates[idx] for idx in idxs_users]
        print("Selected clients:", chosen_users)
        client_selection_history.append(chosen_users)
        client_set |= set(chosen_users)
        if args.wandb:
            wandb.log({"diff_client_num":len(client_set), "Round": iter+1})

        # compute total training data distribution (by classes)
        # class_tally = [0 for c in range(args.num_classes)]
        # for user_id in chosen_users:
        #     user_train_dataset = DatasetSplit(dataset_train, dict_users[user_id])
        #     for i in range(len(user_train_dataset)):
        #         image, label = user_train_dataset[i]
        #         class_tally[label] += 1
        # print("Total training data by class: {}, mean {}, std {}".format(class_tally, sum(class_tally)/len(class_tally), pstdev(class_tally)))


        for idx in chosen_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
            model_copy = type(net_glob.module)(**model_args) # get a new instance
            model_copy = nn.DataParallel(model_copy)
            model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff
            w, loss, trained_data_size, activity = local.train(net=model_copy.to(args.device))
            w_locals_all.append(copy.deepcopy(w))
            loss_locals_all.append(copy.deepcopy(loss))
            trained_data_size_all.append(trained_data_size)

        # for idx in range(len(chosen_users)):
        #     w_locals_selected.append(copy.deepcopy(w_locals_all[idx]))
        #     loss_locals_selected.append(copy.deepcopy(loss_locals_all[idx]))
        
        # model_dev_list = model_deviation(w_locals_all, net_glob.state_dict())
        # ms_model_deviation.append(model_dev_list)

        # update global weights
        w_glob = fl.FedAvg(w_locals_all, w_init = net_glob.state_dict())
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
 
        loss_avg = sum(loss_locals_all) / len(loss_locals_all)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)
        if args.wandb:
            wandb.log({"client_avg_loss":loss_avg, "Round": iter+1})

        if iter % args.eval_every == 0:
            # testing
            net_glob.eval()
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            print("Round {:d}, Training accuracy: {:.2f}".format(iter, acc_train))
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Round {:d}, Testing accuracy: {:.2f}".format(iter, acc_test))

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

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    print("Final Training accuracy: {:.2f}".format(acc_train))
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Final Testing accuracy: {:.2f}".format(acc_test))

    if args.wandb:
        wandb.log({"server_train_loss": loss_train, "server_test_loss": loss_test, 
                    "server_train_acc": acc_train, "server_test_acc": acc_test, "Round": args.epochs})

    # Add metrics to store
    ms_acc_train_list.append(acc_train)
    ms_acc_test_list.append(acc_test)
    ms_loss_train_list.append(loss_train)
    ms_loss_test_list.append(loss_test)

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

    # Save client selection history and count total number of clients that has been chosen at least once
    f = open("./{}/client_selection_history.txt".format(args.result_dir), "w")
    f.write("Client selection history\n")
    s = set()
    for i in range(len(client_selection_history)):
        f.write("Round {}, selected: {} \n".format(i+1, client_selection_history[i]))
    f.write("Selected {} different clients\n".format(len(client_set)))
    for c in dropped_clients:
        f.write("Round {}, never again choosing {}\n".format(c[0], c[1]))
    f.close()
