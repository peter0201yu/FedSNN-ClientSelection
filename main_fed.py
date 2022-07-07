#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

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

from utils.sampling import mnist_iid, mnist_non_iid, cifar_iid, cifar_non_iid, mnist_dvs_iid, mnist_dvs_non_iid, nmnist_iid, nmnist_non_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedLearn
from models.Fed import model_deviation
from models.test import test_img
import models.vgg as ann_models
import models.resnet as resnet_models
import models.vgg_spiking_bntt as snn_models_bntt
from models.simple_conv_cf10 import Simple_CF10_BNTT
from models.simple_conv_mnist import Simple_Mnist_BNTT, Simple_Mnist_NoBNTT, Simple_Mnist_BNTT_Rate
import models.client_selection as client_selection

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
    elif args.dataset == 'N-MNIST':
        dataset_train, dataset_test = nmnist_train_test("nmnist/data")
        if args.iid:
            dict_users = nmnist_iid(dataset_train, args.num_users)
        else:
            dict_users = nmnist_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == 'MNIST':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_non_iid(dataset_train, args.num_classes, args.num_users, args.alpha)
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

    print("dict_users: ", [len(ds) for ds in dict_users.values()])


    # build model
    model_args = {'args': args}
    if args.model[0:3].lower() == 'vgg':
        if args.snn:
            model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps}
            net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).cuda()
        else:
            model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': 3, 'dropout': args.dropout}
            net_glob = ann_models.VGG(**model_args).cuda()
    elif args.model[0:6].lower() == 'resnet':
        if args.snn:
            pass
        else:
            model_args = {'num_cls': args.num_classes}
            net_glob = resnet_models.Network(**model_args).cuda()
    elif args.model == 'simple':
        model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps, 'img_size': args.img_size}
        if args.dataset == 'MNIST' or args.dataset == 'EMNIST':
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
    print(net_glob)

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
    ms_num_client_list, ms_tot_comm_cost_list, ms_avg_comm_cost_list, ms_max_comm_cost_list = [], [], [], []
    ms_tot_nz_grad_list, ms_avg_nz_grad_list, ms_max_nz_grad_list = [], [], []
    # ms_model_deviation = []

    # testing
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Initial Training accuracy: {:.2f}".format(acc_train))
    # print("Initial Testing accuracy: {:.2f}".format(acc_test))
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

    for iter in range(args.epochs):
        print("Learning rate: ", args.lr)
        net_glob.train()
        w_locals_selected, loss_locals_selected = [], []
        w_locals_all, loss_locals_all = [], []
        trained_data_size_all = []
        
        candidates = [idx for idx in range(args.num_users) if len(dict_users[idx]) > args.bs]
        if args.candidate_selection == "random":
            print("Selecting candidates randomly")
            candidates = np.random.choice(range(args.num_users), size=num_candidates, replace=False)
        elif args.candidate_selection == "loop":
            print("Selecting candidates in a loop")
            start = (num_candidates * round) % num_users
            end = min(num_users, start + num_candidates)
            candidates = [i for i in range(start, end)]
        elif args.candidate_selection == "data_amount":
            print("Selecting candidates based on amount of data")
            candidates = np.random.choice(range(args.num_users), size=num_candidates, replace=False, p=data_probs)
        elif args.candidate_selection == "reduce_collision":
            print("Selecting candidates based on prob to reduce collision")
            if chosen_candidates is not None:
                for c in chosen_candidates:
                    data_probs[c] /= args.gamma
                for c in chosen_users:
                    data_probs[c] /= args.gamma
                sum_prob = sum(data_probs)
                data_probs = [prob / sum_prob for prob in data_probs]
            candidates = np.random.choice(range(args.num_users), size=num_candidates, replace=False, p=data_probs)
        elif args.candidate_selection == "avoid_bad":
            print("Selecting candidates by avoiding ones that causes decrease in train acc")
            if chosen_candidates is not None and ms_acc_test_list[-1] < ms_acc_test_list[-2]:
                if iter >= 10 and ms_acc_train_list[-2] - ms_acc_train_list[-1] > 5:
                    # 5% drop in accuracy after 10 epochs, do not train again with these clients
                    for c in chosen_users:
                        data_probs[c] = 0
                    dropped_clients.append((iter+1, chosen_users))
                # avoid choosing them again
                for c in chosen_candidates:
                    data_probs[c] /= args.gamma
                for c in chosen_users:
                    data_probs[c] /= args.gamma
                sum_prob = sum(data_probs)
                data_probs = [prob / sum_prob for prob in data_probs]
            candidates = np.random.choice(range(args.num_users), size=num_candidates, replace=False, p=data_probs)
        elif args.candidate_selection == "keep_good_avoid_bad":
            print("Selecting candidates by continuing with ones that cause increase in train acc")
            if chosen_candidates is not None and ms_acc_test_list[-1] - ms_acc_test_list[-2] > 3:
                for c in chosen_candidates:
                    data_probs[c] *= args.gamma
                # for c in chosen_users:
                #     data_probs[c] *= args.gamma
                sum_prob = sum(data_probs)
                data_probs = [prob / sum_prob for prob in data_probs]
            elif chosen_candidates is not None:
                if iter >= 10 and ms_acc_train_list[-2] - ms_acc_train_list[-1] > 5:
                    # 5% drop in accuracy after 10 epochs, do not train again with these clients
                    for c in chosen_users:
                        data_probs[c] = 0
                    dropped_clients.append((iter+1, chosen_users))
                # avoid choosing them again
                for c in chosen_candidates:
                    data_probs[c] /= args.gamma
                for c in chosen_users:
                    data_probs[c] /= args.gamma
                sum_prob = sum(data_probs)
                data_probs = [prob / sum_prob for prob in data_probs]
        
            candidates = np.random.choice(range(args.num_users), size=num_candidates, replace=False, p=data_probs)

        chosen_candidates = copy.deepcopy(candidates)
        print("candidate clients: ", candidates)
        
        # for idx in idxs_users:
        # Do local update in all the clients # Not required (local updates in only the selected clients is enough) for normal experiments but neeeded for model deviation analysis
        for idx in candidates:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
            model_copy = type(net_glob.module)(**model_args) # get a new instance
            model_copy = nn.DataParallel(model_copy)
            model_copy.load_state_dict(net_glob.state_dict()) # copy weights and stuff
            w, loss, trained_data_size = local.train(net=model_copy.to(args.device))
            w_locals_all.append(copy.deepcopy(w))
            loss_locals_all.append(copy.deepcopy(loss))
            trained_data_size_all.append(trained_data_size)

        # print("local loss: ", loss_locals_all)
        # print("training data distribution: ", trained_data_size_all)
        
        if args.client_selection == "random":
            idxs_users = client_selection.random(len(candidates), m)
        elif args.client_selection == "biggest_train_loss":
            idxs_users = client_selection.biggest_loss(loss_locals_all, len(candidates), m)
        elif args.client_selection == "grad_diversity":
            delta_w_locals_all = []
            w_init = net_glob.state_dict()
            for i in range(len(w_locals_all)):
                delta_w = {}
                for k in w_init.keys():
                    delta_w[k] = w_locals_all[i][k] - w_init[k]
                delta_w_locals_all.append(delta_w)
            idxs_users = client_selection.grad_diversity(delta_w_locals_all, len(candidates), m)
        elif args.client_selection == "update_norm" or args.client_selection == "update_norm_rescale":
            delta_w_locals_all = []
            w_init = net_glob.state_dict()
            for i in range(len(w_locals_all)):
                delta_w = {}
                for k in w_init.keys():
                    delta_w[k] = w_locals_all[i][k] - w_init[k]
                delta_w_locals_all.append(delta_w)
            
            if args.client_selection == "update_norm":
                idxs_users, delta_w_locals_all_rescaled = client_selection.update_norm(delta_w_locals_all, trained_data_size_all, len(candidates), m)
            else:
                idxs_users, delta_w_locals_all_rescaled = client_selection.update_norm(delta_w_locals_all, trained_data_size_all, len(candidates), m, rescale=(True if iter % 5 != 0 else False) )
                # update new weights:
                for i in range(len(w_locals_all)):
                    for k in w_init.keys():
                        w_locals_all[i][k] = w_init[k] + delta_w_locals_all_rescaled[i][k]
        elif args.client_selection == "hybrid":
            if iter%2:
                delta_w_locals_all = []
                w_init = net_glob.state_dict()
                for i in range(len(w_locals_all)):
                    delta_w = {}
                    for k in w_init.keys():
                        delta_w[k] = w_locals_all[i][k] - w_init[k]
                    delta_w_locals_all.append(delta_w)
                idxs_users = client_selection.grad_diversity(delta_w_locals_all, len(candidates), m)
            else:
                idxs_users = client_selection.biggest_loss(loss_locals_all, len(candidates), m)

        # idxs_users gives the client's index in the candidates list, need to convert
        chosen_users = [candidates[idx] for idx in idxs_users]
        print("Selected clients:", chosen_users)
        client_selection_history.append(chosen_users)
        client_set |= set(chosen_users)
        if args.wandb:
            wandb.log({"diff_client_num":len(client_set), "Round": iter+1})
        

        for idx in idxs_users:
            w_locals_selected.append(copy.deepcopy(w_locals_all[idx]))
            loss_locals_selected.append(copy.deepcopy(loss_locals_all[idx]))
        
        # model_dev_list = model_deviation(w_locals_all, net_glob.state_dict())
        # ms_model_deviation.append(model_dev_list)

        # update global weights
        w_glob = fl.FedAvg(w_locals_selected, w_init = net_glob.state_dict())
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
 
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
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
