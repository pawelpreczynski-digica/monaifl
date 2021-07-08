""" server
    - send and recieve models
    - send and receive hyper-parameters
    - send and receive metrics
    - save global models
    - load global models

aggregator
    - federated aggregation
    - distributed aggregation

coordinator
    - contains support utility classes for aggregator

computeplan
    - step-by-step plan to execute FL strategies across the training network

client
    - send and receive model
    - send and receive hyper-parameters
    - send and receive metrics
    - save and load models

reporter
    - contain support and utility classes for client and compute plan

Common 
    - communication models (proto file and client and server side logic for communication)

Network:
    - Contains Node logic to defin training graphs and execute compute plans across the training network """

import torch as t
from client import Client as clt

class Checkpoint():
    def __init__(self):
        self.weights = None
        self.weight_diff = None
        self.metric_accuracy = None
        self.model = None
        self.encrypt_context = None

class MonaiData(Checkpoint):
    def __init__(self):
        self.data = None    
        self.data_bytes = None

    def send(self):
        cp = Checkpoint()
        cp.weights = self.weights
        cp.weight_diff = self.weight_diff
        cp.metric_accuracy = self.metric_accuracy
        # cp.model = self.model
        # cp.encrypt_context = self.encrypt_context

        t.save(self.data_bytes, cp)
        clt.send(self.data_bytes)       

    def recv(self):
        cp = clt.recv()
        self.data = t.load(cp)
        print(cp)


class Mapping(dict):

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    # def __unicode__(self):
    #     return unicode(repr(self.__dict__))


o = Mapping()
o.foo = "bar"
o['lumberjack'] = 'foo'
o.update({'a': 'b'}, c=44)
print(o['lumberjack'])
print(o)


# md = MonaiData()  
# md.weights = t.Tensor([[1, 2, 3], [4, 5, 6]])
# md.weight_diff = md.weights-1
# md.metric_accuracy = 93

# md.send()
# md.recv()

