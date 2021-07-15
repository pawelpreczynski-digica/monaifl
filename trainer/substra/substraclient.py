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
#        self.fl_response= None
        self.data = None
        self.model = None
        self.optimizer = None

    def request(self, model, optim, data):
        self.data = data
        print(self.data.keys())
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        print(size)
        opts = [('grpc.max_receive_message_length', size*2), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', size*2)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.ParamTransfer(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)
        self.model = model
        response = t.load(response_bytes)
        #print(response)
        self.model.load_state_dict(response['weights'])
        self.optimizer = optim
        self.model.eval()
        #print(self.model.state_dict())
        t.save(self.model.state_dict(), modelFile)
        print("Model saved... at: "+ modelFile)
     
    def response(self):
        print("Connecting and recveing data")


