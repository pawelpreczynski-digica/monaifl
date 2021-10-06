from pathlib import Path
cwd = str(Path.cwd())
print(cwd)
import os
import sys
sys.path.append('.')

import asyncio
from concurrent import futures
from io import BytesIO
import grpc
from common import monaifl_pb2_grpc as monaifl_pb2_grpc
from common.monaifl_pb2 import ParamsResponse
from common.utils import Mapping
import torch as t
import copy
import subprocess
from flnode.start_pipeline import instantiateMonaiAlgo
import logging
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)


modelName = "monai-test.pth.tar"

headmodelpath = os.path.join(cwd, "save","models","node2","head")
headModelFile = os.path.join(headmodelpath, modelName)

trunkmodelpath = os.path.join(cwd, "save","models","node2","trunk")
trunkModelFile = os.path.join(trunkmodelpath, modelName)

w_loc = []
request_data = Mapping()
ma, class_names = instantiateMonaiAlgo(0.4, 0.5)

class MonaiFLService(monaifl_pb2_grpc.MonaiFLServiceServicer):
            
    
    def ModelTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        t.save(request_data, headModelFile)
        if os.path.isfile(headModelFile):
            request_data.update(reply="Model received")
            logger.info(f"Global model saved at: {headModelFile}")
            logger.info("FL node is ready for training and waiting for training configurations")
        else:
            request_data.update(reply="Error while receiving the model")
            logger.error("FL node is not ready for training")
        
        logger.info("Returning answer to the Central Hub...")
        buffer = BytesIO()
        t.save(request_data['reply'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def MessageTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        logger.info('Received training configurations')
        logger.info(f"Local epochs to run: {request_data['epochs']}")
        # training and checkpoints
        logger.info("Starting training...")
        checkpoint = Mapping()
        checkpoint = ma.train()
        logger.info("Saving trained local model...")
        t.save(checkpoint, trunkModelFile)
        logger.info(f"Local model saved at: {trunkModelFile}")
        
        logger.info("Sending training completed message to the the Central Hub...")
        buffer = BytesIO()
        request_data.update(reply="Training started")
        t.save(request_data['reply'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def TrainingStatus(self, request, context):
        logger.info("Received the training status request")
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        
        if os.path.isfile(trunkModelFile):
            request_data.update(reply="Training completed")
            logger.info("Training status: completed")
        else:
            request_data.update(reply="Training in progress")
            logger.info("Training status: in progress")

        logger.info("Sending training status to the Central Hub...")
        buffer = BytesIO()
        t.save(request_data['reply'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def TrainedModel(self, request, context):
        buffer = BytesIO()
        if os.path.isfile(trunkModelFile):
                logger.info(f"sending trained model {trunkModelFile} to the Central Hub...") 
                checkpoint = t.load(trunkModelFile)
                t.save(checkpoint, buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def ReportTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        logger.info('Received test request')

        response_data = Mapping()
        response_data = ma.predict(class_names, headModelFile)

        logger.info("Sending test report to the Central Hub...")       
        buffer = BytesIO()
        t.save(response_data['report'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def StopMessage(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        logger.info('Received stop request')

        logger.info("Sending stopping status to the Central Hub...")
        buffer = BytesIO()
        response_data = Mapping()
        response_data.update(reply="Stopping")
        t.save(response_data, buffer)

        logger.info('Node stopping... (Not implemented yet)')   
        return ParamsResponse(para_response=buffer.getvalue())

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=[
               ('grpc.max_send_message_length', 1000*1024*1024),
               ('grpc.max_receive_message_length', 1000*1024*1024)])
    monaifl_pb2_grpc.add_MonaiFLServiceServicer_to_server(
        MonaiFLService(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    logger.info("Trainer is up and waiting for training configurations...")
    server.wait_for_termination()
    #print("server stopped")

if __name__ == "__main__":
    serve()
