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

headmodelpath = os.path.join(cwd, "save","models","node","head")
headModelFile = os.path.join(headmodelpath, modelName)

trunkmodelpath = os.path.join(cwd, "save","models","node","trunk")
trunkModelFile = os.path.join(trunkmodelpath, modelName)

w_loc = []
request_data = Mapping()
ma, class_names = instantiateMonaiAlgo(0.4, 0.5)

class MonaiFLService(monaifl_pb2_grpc.MonaiFLServiceServicer):
    
    def __init__(self):
        self.model = None
        
    
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
        print(context)
        # training and checkpoints
        logger.info("Starting training...")
        checkpoint = Mapping()
        checkpoint = ma.train()
        logger.info("Saving trained local model...")
        t.save(checkpoint, trunkModelFile)
        #train_instruction = 'python start_pipeline.py'
        #subprocess.Popen(['python', 'flnode/start_pipeline.py', '0'], close_fds=True)
        #subprocess.Popen(train_instruction, close_fds=True)
#        b = os.popen(train_instruction)
#        print(b)

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
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("Trainer is up and waiting for training configurations...")
    server.wait_for_termination()
    #print("server stopped")

if __name__ == "__main__":
    serve()







####extra code
# def ModelTransfer(self, request, context):
    #     request_bytes = BytesIO(request.para_request)
    #     request_data = t.load(request_bytes, map_location='cpu')
    #     print('Received Model Request: ', request_data.keys())   
    #     buffer = BytesIO()
    #     if request_data['id'] in whitelist:
    #         print(request_data['id'])
    #         self.model = request_data['model']
    #         if os.path.isfile(modelFile):
    #             print("sending model...") 
    #             print(modelFile)
    #             checkpoint = t.load(modelFile)
    #             t.save(checkpoint['weights'], buffer)
    #         else:
    #             print("initial model does not exist, initializing and sending a new one...")
    #             t.save(self.model.state_dict(), buffer)
    #     else:
    #         print("Please contact admin for permissions...")
    #     return ParamsResponse(para_response=buffer.getvalue())
    # def ParamTransfer(self, request, context):
    #     epochs = 0
    #     w_glob = list() 
    #     optimizer = list()  
    #     request_bytes = BytesIO(request.para_request)
    #     request_data = t.load(request_bytes, map_location='cpu')
    #     print('Received model updates (keys): ', request_data.keys())
      
    #     print("Aggregating model on the server...")     
        
    #     for key in request_data.keys():
    #         if key == 'epoch':
    #             epochs = request_data['epoch']
    #             print("Best Epoch at Client: " + str(request_data['epoch']) )
    #         elif key == 'weights':
    #             w = request_data['weights']
    #             print("Copying weights...")
    #             w_loc.append(copy.deepcopy(w))
    #             print("Aggregating weights...")
    #             #w_glob = FedAvg(w_loc)
    #         elif key == 'optimizer':
    #             optimizer = request_data['optimizer']
    #         elif key == 'metric':
    #             epochs = request_data['metric']
    #             print("Best metric at Client: " + str(request_data['metric']) )
            
    #         else:
    #             print('Server does not recognized the sent data')
    #     buffer = BytesIO()
    #     checkpoint = {'epoch': epochs,
    #         'weights': w_glob,
    #         'optimizer': optimizer}
    #     t.save(checkpoint, modelFile)
    #     t.save(checkpoint, buffer)
    #     print("Returning Checkpoint...") 
    #     return ParamsResponse(para_response=buffer.getvalue())
   
    # # def ParamTransfer(self, request, context):
    #     epochs = 0
    #     w_glob = list() 
    #     optimizer = list()  
    #     request_bytes = BytesIO(request.para_request)
    #     request_data = t.load(request_bytes, map_location='cpu')
    #     print('Received model updates (keys): ', request_data.keys())
      
    #     print("Aggregating model on the server...")     
        
    #     for key in request_data.keys():
    #         if key == 'epoch':
    #             epochs = request_data['epoch']
    #             print("Best Epoch at Client: " + str(request_data['epoch']) )
    #         elif key == 'weights':
    #             w = request_data['weights']
    #             print("Copying weights...")
    #             w_loc.append(copy.deepcopy(w))
    #             print("Aggregating weights...")
    #             #w_glob = FedAvg(w_loc)
    #         elif key == 'optimizer':
    #             optimizer = request_data['optimizer']
    #         elif key == 'metric':
    #             epochs = request_data['metric']
    #             print("Best metric at Client: " + str(request_data['metric']) )
            
    #         else:
    #             print('Server does not recognized the sent data')
    #     buffer = BytesIO()
    #     checkpoint = {'epoch': epochs,
    #         'weights': w_glob,
    #         'optimizer': optimizer}
    #     t.save(checkpoint, modelFile)
    #     t.save(checkpoint, buffer)
    #     print("Returning Checkpoint...") 
    #     return ParamsResponse(para_response=buffer.getvalue())
 



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


        # request_bytes = BytesIO(request.para_request)
        # request_data = t.load(request_bytes, map_location='cpu')
        # print('Received Model Report: ', request_data.keys())   
        # buffer = BytesIO()
        # if request_data['report']:
        #     print(request_data['report'])
        #     request_data.update(reply="Thanks for reporting test statistics")
        # else:
        #     print("No test statistics were reported...")
        #     request_data.update(reply="Server was expecting test statistics but nothing received yet")
        # t.save(request_data['reply'], buffer)
        # return ParamsResponse(para_response=buffer.getvalue())