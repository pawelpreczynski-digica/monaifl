from pathlib import Path
cwd = str(Path.cwd())
print(cwd)

import os
import sys
sys.path.append('.')

import grpc
from common.monaifl_pb2_grpc import MonaiFLServiceStub
from common.monaifl_pb2 import ParamsRequest
from io import BytesIO
from common.utils import Mapping
import numpy as np
import torch as t
import io
import os
import copy
from hub.coordinator import FedAvg

#modelpath = os.path.join(cwd, "save","models","client")
#modelName = 'monai-test.pth.tar'


modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"

modelFile = os.path.join(modelpath, modelName)
w_loc = []
request_data = Mapping()
whitelist = ["localhost:50051", "client2"]
w_glob = list() 
class Client():
    def __init__(self, id, address):
        self.id = id
        self.address = address
        self.client = None
        self.fl_request = None
#       self.fl_response= None
        self.data = None
        self.model = None
        self.optimizer = None
        self.modelFile = os.path.join(modelpath, modelName)

    

    def bootstrap(self):
        print("Connecting with client..."+ self.address)
        buffer = BytesIO()
        if self.address in whitelist:
            print(self.address + " is whitelisted client")
            #self.model = request_data['model']
            if os.path.isfile(modelFile):
                print("sending model...") 
                print(modelFile)
                checkpoint = t.load(modelFile)
                t.save(checkpoint, buffer)
            else:
                print("initial model does not exist, initializing and sending a new one...")
                t.save(self.model.state_dict(), buffer)
            size = buffer.getbuffer().nbytes
            opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
            self.channel = grpc.insecure_channel(self.address, options = opts)
            client = MonaiFLServiceStub(self.channel)
            self.fl_request = ParamsRequest(para_request=buffer.getvalue())
            fl_response = client.ModelTransfer(self.fl_request)
            response_bytes = BytesIO(fl_response.para_response)
            response_data = t.load(response_bytes, map_location='cpu')
            print('Model Received?: ', response_data)   
        else:
            print("Please contact admin for permissions...")
        #return ParamsResponse(para_response=buffer.getvalue())

    def train(self, epochs):
        self.data = {"epochs": epochs}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.MessageTransfer(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        print('Training Started?: ', response_data)   

    def status(self):
        self.data = {"check": 'check'}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.TrainingStatus(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        return response_data
  
    def gather(self):
        self.data = {"id": "server"}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.TrainedModel(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        print('Received trained model')
        print("Copying weights...")
        w_loc.append(copy.deepcopy(response_data))
        w_glob = FedAvg(w_loc)
        buffer = BytesIO()
        t.save(w_glob, modelFile)
        #print(w_glob)
    
    def test(self):
        self.data={"id":"server"}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.ReportTransfer(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        print(response_data)
    
    def stop(self):
        self.data={"stop":"yes"}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        self.fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.StopMessage(self.fl_request)
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        print(response_data['reply'])
  
  #extra code
    #     self.data = {"id": self.id, "model": model}
    #     buffer = BytesIO()
    #     t.save(self.data, buffer)
    #     size = buffer.getbuffer().nbytes
    #     opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
    #     self.channel = grpc.insecure_channel(self.address, options = opts)
    #     client = MonaiFLServiceStub(self.channel)
    #     self.fl_request = ParamsRequest(para_request=buffer.getvalue())
    #     fl_response = client.ModelTransfer(self.fl_request)
    #     response_bytes = BytesIO(fl_response.para_response)
    #     self.model = model
    #     self.response = t.load(response_bytes)
    #     print("Model received...")
    #     self.model.load_state_dict(self.response, strict=False)
    #     self.optimizer = optim
    #     self.model.eval()
    #     t.save(self.model.state_dict(), self.modelFile)
    #     print("Model saved... at: "+ self.modelFile)


    # def aggregate(self, model, optim, data):
    #     print("sending model checkpoint for aggregation...")
    #     self.data = data
    #     # print(self.data.keys())
    #     buffer = BytesIO()
    #     t.save(self.data, buffer)
    #     size = buffer.getbuffer().nbytes
    #     opts = [('grpc.max_receive_message_length', size*2), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', size*2)]
    #     self.channel = grpc.insecure_channel(self.address, options = opts)
    #     client = MonaiFLServiceStub(self.channel)
    #     self.fl_request = ParamsRequest(para_request=buffer.getvalue())
    #     fl_response = client.ParamTransfer(self.fl_request)
    #     response_bytes = BytesIO(fl_response.para_response)
    #     self.model = model
    #     response = t.load(response_bytes)
    #     self.model.load_state_dict(response['weights'])
    #     self.optimizer = optim
    #     self.model.eval()
    #     t.save(self.model.state_dict(), self.modelFile)
    #     print("Model saved... at: "+ self.modelFile)

    # def report(self, data):
    #     print("sending client test report to the server...")
    #     self.data = data
    #     print(self.data['report'])
    #     buffer = BytesIO()
    #     t.save(self.data, buffer)
    #     size = buffer.getbuffer().nbytes
    #     opts = [('grpc.max_receive_message_length', size*2), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', size*2)]
    #     self.channel = grpc.insecure_channel(self.address, options = opts)
    #     client = MonaiFLServiceStub(self.channel)
    #     self.fl_request = ParamsRequest(para_request=buffer.getvalue())
    #     fl_response = client.ReportTransfer(self.fl_request)
    #     response_bytes = BytesIO(fl_response.para_response)
    #     response_data = t.load(response_bytes, map_location='cpu')
    #     print('Received Server Report: ', response_data)   
