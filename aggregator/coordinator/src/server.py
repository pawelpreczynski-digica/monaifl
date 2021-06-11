import sys
sys.path.insert(1, '/.../monaifl/common')

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
    def ParamTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes)
        #print('Received Tensor from Client: ', request_data)
        print("Computing Operations on Tensors")     
        w = request_data
        w_loc.append(copy.deepcopy(w))
        w_glob = FedAvg(w_loc)
        print(w_glob)
        buffer = BytesIO()
        t.save(w_glob, buffer)
        print(buffer)
     #   setGlobalParameters(buffer)
        return ParamsResponse(para_response=buffer.getvalue())
 
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    monaifl_pb2_grpc.add_MonaiFLServiceServicer_to_server(
        MonaiFLService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Waiting for client tensors...")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()