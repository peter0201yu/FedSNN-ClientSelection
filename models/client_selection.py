import numpy as np
import pandas as pd
import copy
import torch

def random(num_users, num_selected):
    print("Selecting clients randomly")
    return np.random.choice(range(num_users), num_selected, replace=False)

def biggest_loss(local_losses, num_users, num_selected):
    print("Selecting clients by biggest loss")
    if local_losses == []:
        return random(num_users, num_selected)
    ret = sorted(range(len(local_losses)), key=lambda x: local_losses[x])
    return ret[(-1)*num_selected:]

def grad_diversity(delta_w_locals_all, num_users, num_selected):
    print("Selecting clients by gradient diversity")
    if delta_w_locals_all == []:
        return random(num_users, num_selected)
    
    # Greedy algorithm
    chosen_users = []
    unchosen_users = [i for i in range(num_users)]
    for i in range(num_selected):
        # print("Unchosen users: ", unchosen_users)
        # Find client that gives largest marginal gain
        min_diff = -1
        chosen = -1
        for j in unchosen_users:
            l = copy.deepcopy(chosen_users)
            l.append(j)
            diff = sum_grad_diff(delta_w_locals_all, num_users, l)
            if min_diff < 0 or diff < min_diff:
                min_diff = diff
                chosen = j
        chosen_users.append(chosen)
        unchosen_users.remove(chosen)

    # print(chosen_users)
    return chosen_users

# Helper for grad_diversity
# Computes the difference between the weighted sum of the chosen gradients
# and the sum of all gradients
def sum_grad_diff(delta_w_locals_all, num_users, l):
    sum_diff = 0

    for i in range(num_users):
        # find user in l with closest gradient to delta_w_locals_all[i]
        min_diff = -1
        for j in l:
            # Find gradient difference between users
            diff = 0
            for k in delta_w_locals_all[0].keys():
                diff += torch.norm(delta_w_locals_all[i][k].float() - delta_w_locals_all[j][k].float())
            if min_diff < 0 or diff < min_diff:
                min_diff = diff
        sum_diff += min_diff
    
    return sum_diff

