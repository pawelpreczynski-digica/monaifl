import copy
import numpy as np

def FedAvg(w):
    w_avg = copy.deepcopy(w)
    for k in w_avg.keys():
        for i in range(1, len(w)):
            print(i,k)
            #print(w[i][k])
            w_avg[k] += w[i][k]
        w_avg[k] = np.divide(w_avg[k], len(w))
    return w_avg
