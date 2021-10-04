from pathlib import Path
cwd = str(Path.cwd())
print(cwd)

import sys
sys.path.append('.')
import os
from substraclient import Client
import time
import logging
import concurrent.futures
import copy
import time
from common.utils import Mapping
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
from io import BytesIO
import torch as t
from hub.coordinator import FedAvg
from flnode.pipeline.monaialgo import MonaiAlgo
from monai.networks.nets import densenet121


#ma = MonaiAlgo()
# model initiliatization
model = densenet121(spatial_dims=2, in_channels=1, out_channels=6)#.to(device)
# model loss function
#ma.loss_function = t.nn.CrossEntropyLoss()
# model optimizer
optimizer = t.optim.Adam(model.parameters(), 1e-5)

modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"

modelFile = os.path.join(modelpath, modelName)
w_loc = []
#checkpoints = Mapping()

nodes ={  1: {'address': 'localhost:50051', 'cp':{}},
          2: {'address': 'localhost:50052', 'cp':{}},
       }

clients = (
    Client(nodes[1]['address']),
    Client(nodes[2]['address'])
)

def train_plan(client):
    # spreading model to nodes
    client.bootstrap(model, optimizer)

    # initializing training on nodes
    client.train(epochs='1')

def aggregate():
    for client in clients:
        print(client.address)
        checkpoint = client.gather()
        for key in checkpoint.keys():
            if key == 'epoch':
                epochs = checkpoint['epoch']
                print("Best Epoch at Client: " + str(checkpoint['epoch']) )
            elif key == 'weights':
                w = checkpoint['weights']
                print("Copying weights...")
                w_loc.append(copy.deepcopy(w))
                print("Aggregating weights...")
                w_glob = FedAvg(w_loc)
                
            # elif key == 'optimizer':
            #     optimizer = checkpoint['optimizer']
            # elif key == 'metric':
            #     epochs = checkpoint['metric']
            #     print("Best metric at Client: " + str(checkpoint['metric']) )
            
            else:
                print('Server does not recognized the sent data')

        buffer = BytesIO()
        checkpoint = {'epoch': epochs,
            'weights': w_glob,
            #'optimizer': optimizer
            }
        t.save(checkpoint, modelFile)

        print("aggregation completed")

def test_plan(client):
    # testing models on nodes
    client.test()

    # asking nodes to stop
    client.stop()


if __name__ == '__main__':
    logger.info("Central Hub initialized")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(train_plan, clients)    
    
    aggregate()
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(test_plan, clients)    
    
    # all process excuted 
    print("Done!")




#rough code
#     function to check the traing status.
#     """
#     trained = client.status()
#     while (trained != "Training completed"):
#         time.sleep(1)
#         trained = client.status()
#         print(trained)





    # w_glob = list()
    # for client in clients:
    #     print(client.address)
    #     checkpoint = client.gather()
    #     w = checkpoint['weights']
    #     print("Copying weights...")
    #     w_loc.append(copy.deepcopy(w))
        
    # print("Aggregating weights...")
    # w_glob = FedAvg(w_loc)
    # t.save(w_glob, modelFile)
    # print("aggregation completed")

# def gather_plan(clients):
#     print("gathering node weights...")
#     checkpoint = clients.gather()
#     print(checkpoint.keys())
#     #checkpoints.update({'global_weights': FedAvg(checkpoint['weights'])})
#     #cpt = {'id': clients.address, 'cp': checkpoint}
#     #checkpoints.update(cpt)
#     #print(checkpoints.keys())
#     # checked = []
#     # for node in nodes:
#     #     for c in clients:
#     #         if (c.address == nodes[node]['address']) and (c.address not in checked):
#     #             print(c.address)
#     #             checked.append(c.address)
#     #             print(checked)
#     #             checkpoint = c.gather()
#     #             cpt = {'id': c.address, 'cp': checkpoint}
#     #             checkpoints.update(cpt)
#     #             break
#     #     continue
    
# def aggregate():
#     print(checkpoints.items())
#     #print(c.keys())
#     #w_glb = FedAvg(w_loc)
#     #t.save(w_glb, modelFile)
#     print("aggregation completed")

 # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     result = executor.map(gather_plan, clients)    
