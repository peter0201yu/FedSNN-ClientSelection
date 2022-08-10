import numpy as np
import pandas as pd
import copy
import torch

def random(num_users, num_candidates):
    print("Selecting candidates randomly")
    return np.random.choice(range(num_users), size=num_candidates, replace=False)

def loop(num_users, num_candidates, round):
    print("Selecting candidates in a loop")
    start = (num_candidates * round) % num_users
    end = min(num_users, start + num_candidates)
    return [i for i in range(start, end)]

def data_amount(num_users, num_candidates, data_probs):
    print("Selecting candidates based on amount of data")
    return np.random.choice(range(num_users), size=num_candidates, replace=False, p=data_probs)

def reduce_collision(num_users, num_candidates, data_probs, chosen_candidates, chosen_users, gamma):
    print("Selecting candidates based on prob to reduce collision")
    if chosen_candidates is not None:
        for c in chosen_candidates:
            data_probs[c] /= gamma
        for c in chosen_users:
            data_probs[c] /= gamma
        sum_prob = sum(data_probs)
        data_probs = [prob / sum_prob for prob in data_probs]
    return np.random.choice(range(num_users), size=num_candidates, replace=False, p=data_probs)

# This violates the train/test data isolation
# def keep_good_avoid_bad(num_users, num_candidates, data_probs, chosen_candidates, chosen_users, ms_acc_train_list, gamma, round, dropped_clients):
#     print("Selecting candidates by continuing with ones that cause increase in train acc")
#     if chosen_candidates is not None and ms_acc_train_list[-1] - ms_acc_train_list[-2] > 3:
#         for c in chosen_candidates:
#             data_probs[c] *= gamma
#         # for c in chosen_users:
#         #     data_probs[c] *= args.gamma
#         sum_prob = sum(data_probs)
#         data_probs = [prob / sum_prob for prob in data_probs]
#     elif chosen_candidates is not None and ms_acc_train_list[-2] - ms_acc_train_list[-1] > 3:
#         # if round >= 10 and ms_acc_train_list[-2] - ms_acc_train_list[-1] > 5:
#         #     # 5% drop in accuracy after 10 epochs, do not train again with these clients
#         #     for c in chosen_users:
#         #         data_probs[c] = 0
#         #     dropped_clients.append((round+1, chosen_users))
#         # avoid choosing them again
#         for c in chosen_candidates:
#             data_probs[c] /= gamma
#         for c in chosen_users:
#             data_probs[c] /= gamma
#         sum_prob = sum(data_probs)
#         data_probs = [prob / sum_prob for prob in data_probs]

#     return np.random.choice(range(num_users), size=num_candidates, replace=False, p=data_probs)

