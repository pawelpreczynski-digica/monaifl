import copy
import torch
from torch import nn

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

w_locals = []
w = received tensor
w_locals = copy.deepcopy(w)

#w_locals.append(copy.deepcopy(w))
#loss_locals.append(copy.deepcopy(loss))

# update global weights
w_glob = FedAvg(w_locals)


