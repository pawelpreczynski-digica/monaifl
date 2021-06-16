import sys
sys.path.insert(1, '/.../monaifl/common')

import grpc
from monaifl_pb2_grpc import MonaiFLServiceStub
from monaifl_pb2 import ParamsRequest
from io import BytesIO
import numpy as np
import torch as t
import io
import copy


from reporter import getLocalParameters, setLocalParameters

def client():
    channel = grpc.insecure_channel("localhost:50051")
    client = MonaiFLServiceStub(channel)

    y = getLocalParameters()
    if(y):
        print('Preparing Tensor on Client:', y)
        buffer = BytesIO()
        t.save(y, buffer)
        request = ParamsRequest(para_request= buffer.getvalue())#request_bytes.getvalue()})
        response = client.ParamTransfer(request)
        print(response)
        response_bytes = BytesIO(response.para_response)
        setLocalParameters(response_bytes)
    else:
        print("Local model not available yet...")

client()