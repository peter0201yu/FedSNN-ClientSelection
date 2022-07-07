import numpy as np
import pandas as pd
import copy
import torch

def random(num_users, num_selected):
    print("Selecting clients randomly")
    return np.random.choice(range(num_users), num_selected, replace=False)

def biggest_loss(local_losses, num_users, num_selected):
    print("Selecting clients by biggest loss")
    ret = sorted(list(range(num_users)), key=lambda x: local_losses[x], reverse=True)
    # loss_locals_selected = [local_losses[i] for i in ret[:num_selected]]
    # print("Local loss selected:", loss_locals_selected)
    return ret[:num_selected]

def middle_loss(local_losses, num_users, num_selected):
    print("Selecting clients with middle loss")
    ret = sorted(list(range(num_users)), key=lambda x: local_losses[x], reverse=True)
    start = (num_users-num_selected)//2
    return ret[start : start+num_selected]

def smallest_loss(local_losses, num_users, num_selected):
    print("Selecting clients with smallest loss")
    ret = sorted(list(range(num_users)), key=lambda x: local_losses[x], reverse=False)
    return ret[:num_selected]

def mixed_loss(local_losses, num_users, num_selected):
    print("Selecting clients with mixed loss (choosing both big and small)")
    ret = sorted(list(range(num_users)), key=lambda x: local_losses[x], reverse=False)
    interval = num_users // num_selected
    return [ret[interval*i] for i in range(num_selected)]

def grad_diversity(delta_w_locals_all, num_users, num_selected):
    print("Selecting clients by gradient diversity")    
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
        print("Selected client {} with diff {}".format(chosen, min_diff))
        
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
            diff = grad_diff(delta_w_locals_all, i, j)
            if min_diff < 0 or diff < min_diff:
                min_diff = diff
        sum_diff += min_diff
    
    # print("If we choose users {}, get sum diff {}".format(l, sum_diff))
    
    return sum_diff

# Helper for grad_diversity
# Computes the difference between gradients of two clients
def grad_diff(delta_w_locals_all, i, j):
    diff = 0
    for k in delta_w_locals_all[0].keys():
        diff += torch.norm(delta_w_locals_all[i][k].float() - delta_w_locals_all[j][k].float())
    # print("Clients {} and {} has diff {}".format(i, j, diff))
    return diff

def spike_diversity(activities, num_users, num_selected):
    print("Selecting clients by spike activity diversity")
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
            diff = sum_spike_diff(activities, num_users, l)

            if min_diff < 0 or diff < min_diff:
                min_diff = diff
                chosen = j
        print("Selected client {} with diff {}".format(chosen, min_diff))
        
        chosen_users.append(chosen)
        unchosen_users.remove(chosen)

    # print(chosen_users)
    return chosen_users

def sum_spike_diff(activities, num_users, l):
    sum_diff = 0
    cos = torch.nn.CosineSimilarity(dim=0)
    for i in range(num_users):
        # find user in l with closest activity to activities[i]
        min_diff = -1
        active_layer_count = 9
        for j in l:
            print("--------------------------")
            print(activities[i], activities[j])
            # Find spike activity difference between users
            diff = sum([cos(activities[i], activities[j]).item() for layer in range(active_layer_count)])
            print(diff)
            print("--------------------------")
            if min_diff < 0 or diff < min_diff:
                min_diff = diff
        sum_diff += min_diff
    
    # print("If we choose users {}, get sum diff {}".format(l, sum_diff))
    
    return sum_diff

# Pick clients based on norm of updates (Alg1 in Optimal Client Sampling paper)
def update_norm(delta_w_locals_all, trained_data_size_all, num_users, num_selected, rescale=False):
    print("Selecting clients by update norm, rescale = ", rescale)
    
    trained_data_size_total = sum(trained_data_size_all)
    probs = []
    weighted_norms = []
    weights = [trained_data_size_all[i] / trained_data_size_total for i in range(num_users)]
    for i in range(num_users):
        norm = 0
        for k in delta_w_locals_all[0].keys():
            norm += float(torch.linalg.norm(delta_w_locals_all[i][k].float()))
        weighted_norms.append(norm * weights[i]) 
    
    sorted_weighted_norms = sorted(weighted_norms)
    print("sorted weight norms: ", sorted_weighted_norms)
    l = num_users - num_selected + 1
    while l <= num_users and (num_selected + l - num_users <= (sum(sorted_weighted_norms[:l])/sorted_weighted_norms[l-1])):
        l += 1
    l -= 1
    # print("found l: ", l)

    # slight change to the original paper: make sure to choose the clients with prob 1, and then
    # scale down the probs and make a random selection w.r.t probs
    # equation 7 in paper
    for i in range(num_users):
        if l < num_users and weighted_norms[i] >= sorted_weighted_norms[l]:
            probs.append(1)
        else:
            probs.append((num_selected + l - num_users) * (weighted_norms[i]) / sum(sorted_weighted_norms[:l]))
    print("Probilities: ", probs)

    picked_clients = []
    prob_dist = []
    for i in range(num_users):
        if probs[i] == 1:
            picked_clients.append(i)
            prob_dist.append(0)
        else:
            prob_dist.append(probs[i])

    # normalize this
    sum_prob = sum(prob_dist)
    prob_dist = [prob / sum_prob for prob in prob_dist]
    newly_picked = list(np.random.choice(range(num_users), size=num_selected - len(picked_clients), replace=False, p=prob_dist))
    # print("already picked: {}, newly_picked: {}".format(picked_clients, newly_picked))
    picked_clients += newly_picked
    print("picked: ", picked_clients)

    delta_w_locals_all_rescaled = delta_w_locals_all

    if rescale:
        # This rescaling messes up the result ??
        for i in picked_clients:
            for k in delta_w_locals_all[0].keys():
                # rescale by probability and client sampling weight (trained datasize)
                delta_w_locals_all_rescaled[i][k] = torch.mul(delta_w_locals_all_rescaled[i][k].float(), weights[i] / probs[i])

    return picked_clients, delta_w_locals_all_rescaled

# # Pick clients based on approximated norm of updates (Alg2 in Optimal Client Sampling paper)
# def update_norm_approx(delta_w_locals_all, trained_data_size_all, num_users, num_selected, approx_rounds):
#     print("Selecting clients by approximated update norm")
#     if delta_w_locals_all == []:
#         return random(num_users, num_selected)
    
#     probs = []
#     weighted_norms = []
#     for i in range(num_users):
#         norm = 0
#         for k in delta_w_locals_all[0].keys():
#             norm += torch.norm(delta_w_locals_all[i][k].float())
#         weighted_norms.append(norm * trained_data_size_all[i])
    
#     sum_norm = sum(weighted_norms)
#     for i in range(num_users):
#         probs.append(min([num_selected * weighted_norms[i] / sum_norm, 1]))
    
#     for j in range(approx_rounds):
#         master_aggregate = (0, 0)
#         for i in range(num_users):
#             if probs[i] < 1:
#                 master_aggregate += (1, probs[i])
#         C = (num_selected - num_users + master_aggregate[0]) / master_aggregate[1]
#         for i in range(num_users):
#             if probs[i] < 1:
#                 probs[i] = min([C * probs[i], 1])
#         if C <= 1:
#             break

#     print("Probilities: ", probs)
#     print("Sum of probabilities: ", sum(probs))

#     picked_clients = []
#     update_norms_rescaled = delta_w_locals_all
#     for i in range(num_users):
#         if probs[i] >= np.random.uniform(0, 1):
#             picked_clients.append(i)
#             for k in delta_w_locals_all[0].keys():
#                 update_norms_rescaled[i][k].float() *= (trained_data_size_all / probs[i])
#     return picked_clients, update_norms_rescaled
