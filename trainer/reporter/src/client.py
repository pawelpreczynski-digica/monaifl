import grpc
from monaifl_pb2_grpc import MonaiFLServiceStub
from monaifl_pb2 import ParamsRequest
from io import BytesIO
import numpy as np
import torch as t

channel = grpc.insecure_channel("localhost:50051")
client = MonaiFLServiceStub(channel)

x = t.randn(2,10)

print('Preparing Tensor on Client:', x)
request_bytes = BytesIO()
np.save(request_bytes, x, allow_pickle=False)
request = ParamsRequest(paraRequest=request_bytes.getvalue())

response = client.ParamTransfer(request)
response_bytes = BytesIO(response.paraResponse)
response_data = np.load(response_bytes, allow_pickle=False)

print('Received Tensor: ', response_data)
