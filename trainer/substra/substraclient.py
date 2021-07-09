# import sys
# sys.path.insert(0, '/.../common')
# #trainer/reporter/src/client.py

import grpc
from monaifl_pb2_grpc import MonaiFLServiceStub
from monaifl_pb2 import ParamsRequest
from io import BytesIO
import numpy as np
import torch as t
import io
import os
import sys
import copy

#from reporter import getLocalParameters, setLocalParameters

from pathlib import Path
home = str(Path.home())

print(home)
modelpath = os.path.join(home, "monaifl", "save","models","client")
#modelName = 'MNIST-test.pth.tar'
modelName = 'monai-test.pth.tar'
modelFile = os.path.join(modelpath, modelName)

class Client():
    def __init__(self, address):
        self.address = address
        self.client = None
        self.fl_request = None
        self.fl_response= None
        self.data = None
        self.model = None
        self.optimizer = None

    def request(self, model, optim, data):
        self.data = data
        #print(self.data)
        
        if(self.data):
            buffer = BytesIO()
            #print(buffer)
            t.save(self.data, buffer)
            size = sys.getsizeof(buffer)
            options = [('grpc.max_message_length', size)]
            self.channel = grpc.insecure_channel(self.address, options = options)
            self.client = MonaiFLServiceStub(self.channel)
            self.fl_request = ParamsRequest(para_request=buffer.getvalue())
            print(self.fl_request)
            self.fl_response = self.client.ParamTransfer(self.fl_request)
            print(self.fl_response)
            response_bytes = BytesIO(self.fl_response.para_response)
            print(response_bytes)
            self.model = model
            print(self.model)
            self.model.load_state_dict(t.load(response_bytes))
            self.optimizer = optim
            print(self.optimizer)
            self.model.eval()
            t.save(model.state_dict(), modelFile)
            #setLocalParameters(response_bytes)
#        print("creating connection and sending data")
    
    def response(self):
        print("Connecting and recveing data")