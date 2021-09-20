from pathlib import Path
home = str(Path.home())
#print(home)

import os
import sys
sys.path.append('.')

import grpc
from common.monaifl_pb2_grpc import MonaiFLServiceStub
from common.monaifl_pb2 import ParamsRequest
from io import BytesIO
import numpy as np
import torch as t
import io
import os
import copy
#from reporter import getLocalParameters, setLocalParameters

modelpath = os.path.join(home, "monaifl", "save","models","client")
modelName = 'monai-test.pth.tar'
#modelFile = 

class Client():
    def __init__(self, address):
        self.address = address
        self.client = None
        self.fl_request = None
#       self.fl_response= None
        self.data = None
        self.model = None
        self.optimizer = None
        self.modelFile = os.path.join(modelpath, modelName)

     
    def bootstrap(self, model, optim):
        print("Connecting and recveing initial model checkpoint...")
        self.data = {"id":"client1", "model": model}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.ModelTransfer(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)
        self.model = model
        self.response = t.load(response_bytes)
        print("Model received...")
        self.model.load_state_dict(self.response, strict=False)
        self.optimizer = optim
        self.model.eval()
        t.save(self.model.state_dict(), self.modelFile)
        print("Model saved... at: "+ self.modelFile)


    def aggregate(self, model, optim, data):
        print("sending model checkpoint for aggregation...")
        self.data = data
        #print(self.data.keys())
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', size*2), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', size*2)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.ParamTransfer(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)
        self.model = model
        response = t.load(response_bytes)
        self.model.load_state_dict(response['weights'])
        self.optimizer = optim
        self.model.eval()
        t.save(self.model.state_dict(), self.modelFile)
        print("Model saved... at: "+ self.modelFile)

    def report(self, data):
        print("sending client test report to the server...")
        self.data = data
        print(self.data['report'])
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', size*2), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', size*2)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.ReportTransfer(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        print('Received Server Report: ', response_data)   