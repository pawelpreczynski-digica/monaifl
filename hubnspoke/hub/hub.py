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
import torch as t
import os
import copy
import logging
from coordinator import FedAvg

class Stage():
    FEDERATION_INITIALIZATION_STARTED = 'FEDERATION_INITIALIZATION_STARTED'
    FEDERATION_INITIALIZATION_COMPLETED = 'FEDERATION_INITIALIZATION_COMPLETED'
    TRAINING_STARTED = 'TRAINING_STARTED'
    TRAINING_COMPLETED = 'TRAINING_COMPLETED'
    AGGREGATION_STARTED = 'AGGREGATION_STARTED'
    AGGREGATION_COMPLETED = 'AGGREGATION_COMPLETED'
    TESTING_STARTED = 'TESTING_STARTED'
    TESTING_COMPLETED = 'TESTING_COMPLETED'
    UPLOAD_STARTED = 'UPLOAD_STARTED'
    UPLOAD_COMPLETED = 'UPLOAD_COMPLETED'
    FEDERATION_COMPLETED = 'FEDERATION_COMPLETED'


modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"
modelFile = os.path.join(modelpath, modelName)

w_loc = list() 
whitelist = set()

federated_process_logger = logging.getLogger('federated_process') # [%(asctime)s]-[%(model_id)s]-[%(status)s]-[%(trust_name)s]-%(message)s
federated_process_extra = {'model_id': os.environ.get('MODEL_ID'), 'status':'', 'trust_name':''}

node_status_logger = logging.getLogger('node_status') # [%(asctime)s]-[%(model_id)s]-[%(trust_name)s]-%(message)s
node_status_extra = {'model_id': os.environ.get('MODEL_ID'), 'trust_name':''}

logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
        logger.info(f"[{Stage.FEDERATION_INITIALIZATION_STARTED}]-[{self.name}]-bootstrapping...")
        buffer = BytesIO()
        if self.address in whitelist:
            logger.info(f"[{Stage.FEDERATION_INITIALIZATION_STARTED}]-[{self.name}]-fl node is whitelisted")
            if os.path.isfile(modelFile):
                logger.info(f"[{Stage.FEDERATION_INITIALIZATION_STARTED}]-[{self.name}]-buffering the provided initial model {modelFile}...") 
                checkpoint = t.load(modelFile)
                t.save(checkpoint['weights'], buffer)
            else:
                logger.info(f"[{Stage.FEDERATION_INITIALIZATION_STARTED}]-[{self.name}]-initial model does not exist, initializing and buffering a new one...")
                t.save(self.model.state_dict(), buffer)
            size = buffer.getbuffer().nbytes
            
            logger.info(f"[{Stage.FEDERATION_INITIALIZATION_STARTED}]-[{self.name}]-sending the initial model...")
            opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
            self.channel = grpc.insecure_channel(self.address, options = opts)
            client = MonaiFLServiceStub(self.channel)
            fl_request = ParamsRequest(para_request=buffer.getvalue())
            fl_response = client.ModelTransfer(fl_request)

            logger.info(f"[{Stage.FEDERATION_INITIALIZATION_STARTED}]-[{self.name}]-answer received")
            response_bytes = BytesIO(fl_response.para_response)
            response_data = t.load(response_bytes, map_location='cpu')
            logger.info(f"[{Stage.FEDERATION_INITIALIZATION_COMPLETED}]-[{self.name}]-returned status: {response_data}") # Model received OR Error
        else:
            logger.error(f"[{Stage.FEDERATION_INITIALIZATION_COMPLETED}]-[{self.name}]-fl node is not whitelisted. Please contact admin for permissions")

    def train(self, epochs):
        self.data = {"epochs": epochs}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"[{Stage.TRAINING_STARTED}]-[{self.name}]-sending the training request for {epochs} epochs...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.MessageTransfer(fl_request)

        logger.info(f"[{Stage.TRAINING_STARTED}]-[{self.name}]-answer received")
        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        logger.info(f"[{Stage.TRAINING_COMPLETED}]-[{self.name}]-returned status: {response_data}") # Training started 
        return response_data
    
    def status(self):
        self.data = {"check": 'check'}
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"[{self.name}]-checking fl node status...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.NodeStatus(fl_request)

        response_bytes = BytesIO(fl_response.para_response)
        response_data = t.load(response_bytes, map_location='cpu')
        logger.info(f"[{self.name}]-returned status: {response_data}")
        return response_data
  
    def gather(self):
        self.data = {"id": "server"} # useless
        buffer = BytesIO()
        t.save(self.data, buffer)
        size = buffer.getbuffer().nbytes

        logger.info(f"[{Stage.AGGREGATION_STARTED}]-[{self.name}]-Sending the trained model request...")
        opts = [('grpc.max_receive_message_length', 1000*1024*1024), ('grpc.max_send_message_length', size*2), ('grpc.max_message_length', 1000*1024*1024)]
        self.channel = grpc.insecure_channel(self.address, options = opts)
        client = MonaiFLServiceStub(self.channel)
        fl_request = ParamsRequest(para_request=buffer.getvalue())
        fl_response = client.TrainedModel(fl_request)

        logger.info(f"Received the trained model from {self.name}")
        response_bytes = BytesIO(fl_response.para_response)    
        response_data = t.load(response_bytes, map_location='cpu')
        return response_data

    def aggregate(self):
        logger.info(f"Aggregating with Node: {self.name}...") 
        checkpoint = self.gather()
        for k in checkpoint.keys():
            if k == "epoch":
                #epochs = checkpoint['epoch']
                logger.info(f"Best Epoch at Client: {checkpoint['epoch']}...") 
            elif k == "weights":
                w = checkpoint['weights']
                logger.info(f"Copying weights from {self.name}...")
                w_loc.append(copy.deepcopy(w))
                logger.info(f"Aggregating weights from {self.name}...")
                w_glob = FedAvg(w_loc)
            elif k == "metric":
                logger.info(f"Best Metric at Client: {checkpoint['metric']}..." )
            else:
                logger.info(f"Server does not recognized the data sent from {self.name}")
        cpt = {#'epoch': 1, # to be determined
            'weights': w_glob#,
            #'metric': 0 # to be aggregated
            }
        t.save(cpt, modelFile)
        logger.info(f"aggregation with {self.name} completed")
        
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
