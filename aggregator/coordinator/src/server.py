from concurrent import futures
from io import BytesIO
import numpy as np
import grpc
import monaifl_pb2_grpc
from monaifl_pb2 import ParamsResponse
import torch as t

class MonaiFLService(monaifl_pb2_grpc.MonaiFLServiceServicer):
    def ParamTransfer(self, request, context):
        print(request)
        request_bytes = BytesIO(request.paraRequest)
        request_data = np.load(request_bytes, allow_pickle=False)
        print("Received Tensor from Client")
     #   print('Received Tensor from Client: ', request_data)
        print("Computing Operations on Tensors")        
        x = request_data + 5
     #   print('Server Tensor:', x)
        response_bytes = BytesIO()
        np.save(response_bytes, x, allow_pickle=False)
        print("Sent Tensor from Client")
        return ParamsResponse(paraResponse=response_bytes.getvalue())

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