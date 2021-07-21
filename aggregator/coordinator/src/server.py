import os
from concurrent import futures
from io import BytesIO
import numpy as np
import grpc
import monaifl_pb2_grpc
from monaifl_pb2 import ParamsResponse
from utils import Mapping
import torch as t
import copy
from coordinator import FedAvg

from pathlib import Path

home = str(Path.home())
print(home)

modelpath = os.path.join(home, "monaifl", "save","models","server")
modelName = "test-monai.pth.tar"
modelFile = os.path.join(modelpath, modelName)

w_loc = []
request_data = Mapping()
whitelist = ["client1"]

class MonaiFLService(monaifl_pb2_grpc.MonaiFLServiceServicer):
    

    def ParamTransfer(self, request, context):
        epochs = 0
        w_glob = list() 
        optimizer = list()  
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        print('Received Model Updates (keys): ', request_data.keys())
      
        print("Aggregating Model...")     
        
        for key in request_data.keys():
            if key == 'epoch':
                epochs = request_data['epoch']
                print("Best Epoch at Client: " + str(request_data['epoch']) )
            elif key == 'weights':
                w = request_data['weights']
                print("Copying weights...")
                w_loc.append(copy.deepcopy(w))
                print("Aggregating weights...")
                w_glob = FedAvg(w_loc)
            elif key == 'optimizer':
                optimizer = request_data['optimizer']
            elif key == 'metric':
                epochs = request_data['metric']
                print("Best metric at Client: " + str(request_data['metric']) )
            
            else:
                print('Server does not recognized the sent data')
        buffer = BytesIO()
        checkpoint = {'epoch': epochs,
            'weights': w_glob,
            'optimizer': optimizer}
        #print(checkpoint)
        t.save(checkpoint, modelFile)
        t.save(checkpoint, buffer)
        print("Returning Checkpoint...") 
        return ParamsResponse(para_response=buffer.getvalue())
 
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=[
               ('grpc.max_send_message_length', 500*1024*1024),
               ('grpc.max_receive_message_length', 500*1024*1024)])
    monaifl_pb2_grpc.add_MonaiFLServiceServicer_to_server(
        MonaiFLService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Waiting for client tensors...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()