from concurrent import futures
from io import BytesIO
import numpy as np
import grpc
import monaifl_pb2_grpc
from monaifl_pb2 import ParamsResponse

import torch as t
import copy
from coordinator import FedAvg, setGlobalParameters 
import json

w_loc = []
class MonaiFLService(monaifl_pb2_grpc.MonaiFLServiceServicer):
    def __init__(self):
        self.epochs = None
        self.weights = None
        self.optimizer = None 

    def ParamTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes)
        print('Received Model Updates from Client: ', request_data)
        print("Aggregating Model...")     
        if request_data['epoch']:
            self.epochs = request_data['epoch']
            print("Best Epoch at Client: " + request_data['epoch'] )
        elif request_data['weights']:
            w = request_data['weights']
            print(w)
            w_loc.append(copy.deepcopy(w))
            w_glob = FedAvg(w_loc)
            self.weights = w_glob
            print(w_glob)
        elif request_data['optimizer']:
            self.optimizer = request_data['optimizer']
            print("Best Optimizer at Client: " + request_data['optimizer'])
        else:
            print('Server does not recognized the sent data')
        buffer = BytesIO()
        checkpoint = {'epoch': self.epochs,
            'weights': self.weights,
            'optimizer': self.optimizer}
        t.save(checkpoint, buffer)
        #t.save(w_glob, buffer)
        print("Returning Checkpoint...") 
        #print(buffer)
     #   setGlobalParameters(buffer)
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