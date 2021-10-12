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
import torch as t
import os
import copy
import logging
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"

modelFile = os.path.join(modelpath, modelName)
w_loc = list() 
w_glob = list() 
request_data = Mapping()
whitelist = set()


class Client():
    def __init__(self, address, name):
        self.address = address
        whitelist.add(address)
        self.name = name
        self.client = None
        self.data = None
        self.model = None
        self.optimizer = None
        self.modelFile = os.path.join(modelpath, modelName)
        self.loc_weights = None
    

    def bootstrap(self):
        logger.info("bootstrapping with " + self.name)
        buffer = BytesIO()
        if self.address in whitelist:
            logger.info(self.name + " is whitelisted")
            if os.path.isfile(modelFile):
                logger.info(f"buffering the provided initial model {modelFile}...") 
                checkpoint = t.load(modelFile)
                t.save(checkpoint['weights'], buffer)
            else:
                logger.info("initial model does not exist, initializing and buffering a new one...")
                t.save(self.model.state_dict(), buffer)
            size = buffer.getbuffer().nbytes
            
            logger.info(f"sending the initial model to {self.name}...")
            opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
            self.channel = grpc.insecure_channel(self.address, options = opts)
            client = MonaiFLServiceStub(self.channel)
            fl_request = ParamsRequest(para_request=buffer.getvalue())
            fl_response = client.ModelTransfer(fl_request)

            logger.info(f"received the answer from {self.name}")
            response_bytes = BytesIO(fl_response.para_response)
            response_data = t.load(response_bytes, map_location='cpu')
            logger.info(f"{self.name} returned status: {response_data}") # Model received OR Error
        else:
            logger.error(f"{self.name} is not whitelisted. Please contact admin for permissions")

    def train(self, epochs):
        self.data = {"epochs": epochs}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"sending the training request to {self.name} for {epochs} epochs...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.MessageTransfer(fl_request)

        logger.info(f"received the training started ack from {self.name}")
        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        print(response_data)
        logger.info(f"{self.name} returned status: {response_data}") # Training started 
        return response_data
    
    def status(self):
        self.data = {"check": 'check'}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"checking node status: {self.name}...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.NodeStatus(fl_request)

        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        logger.info(f"{self.name} returned status: {response_data}") # Training completed OR Training in progress 
        return response_data
  
    def gather(self):
        self.data = {"id": "server"} # useless
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"Sending the trained model request to {self.name}...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.TrainedModel(fl_request)

        logger.info(f"Received the trained model from {self.name}")
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        return response_data
        
    def test(self):
        buffer = BytesIO()
        checkpoint = t.load(modelFile)
        t.save(checkpoint['weights'], buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"Sending the test request to {self.name}...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.ReportTransfer(fl_request)

        logger.info(f"Received the test report from {self.name}")
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')

        reportName = self.name.replace(' ','') + '.txt'
        reportFile = os.path.join(modelpath, reportName)
        logger.info(f"Writing the test report of {self.name} in {reportFile}...")
        with open(reportFile, 'w') as f:
            f.write(response_data)
    
    def stop(self):
        self.data={"stop":"yes"} # useless
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"Sending the stop message to {self.name}...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.StopMessage(fl_request)

        logger.info(f"Received the node status from {self.name}")
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        logger.info(f"{self.name} returned status: {response_data['reply']}")
