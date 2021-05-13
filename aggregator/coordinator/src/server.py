import grpc
from concurrent import futures
import time
import datacom_pb2_grpc as pb2_grpc
import datacom_pb2 as pb2


class PlainMessageService(pb2_grpc.PlainMessageServicer):

    def __init__(self, *args, **kwargs):
        pass

    def GetServerResponse(self, request, context):

        # get the string from the incoming request
        message = request.message
        result = f'Hello I am up and running received "{message}" message from you'
        result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_PlainMessageServicer_to_server(PlainMessageService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("server is running and waiting for clients...")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()