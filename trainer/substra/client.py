import torch as t

class Client():
    def __init__(self):
        self.id = None
        self.ip = None
        self.port = None
    
    def send(self):
        print("creating connection and sending data")
    
    def recv(self):
        print("Connecting and recveing data")