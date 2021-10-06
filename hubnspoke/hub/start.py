from pathlib import Path
cwd = str(Path.cwd())

import sys
sys.path.append('.')
import os
from hub import Client
import time
import logging
import concurrent.futures
import copy
import time
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
import torch as t
from coordinator import FedAvg

modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"

modelFile = os.path.join(modelpath, modelName)
w_loc = []

nodes ={  1: {'address': 'localhost:50051'},
          2: {'address': 'localhost:50052'},
       }

clients = (Client(nodes[1]['address']), Client(nodes[2]['address']))

def train_plan(client):
    # spreading model to nodes
    client.bootstrap()
    time.sleep(3)
    # initializing training on nodes
    client.train(epochs='1')

def aggregate(client):
    logger.info(f"Aggregating with Node: {client.address}...") 
    checkpoint = client.gather()
    for k in checkpoint.keys():
        if k == "epoch":
            #epochs = checkpoint['epoch']
            logger.info(f"Best Epoch at Client: {checkpoint['epoch']}...") 
        elif k == "weights":
            w = checkpoint['weights']
            logger.info(f"Copying weights from {client.address}...")
            w_loc.append(copy.deepcopy(w))
            logger.info(f"Aggregating weights from {client.address}...")
            w_glob = FedAvg(w_loc)
        elif k == "metric":
            logger.info(f"Best Metric at Client: {checkpoint['metric']}..." )
        else:
            logger.info(f"Server does not recognized the data sent from {client.address}")

    cpt = {#'epoch': 1, # to be determined
            'weights': w_glob#,
            #'metric': 0 # to be aggregated
            }
    t.save(cpt, modelFile)
    logger.info(f"aggregation with {client.address} completed")
    
def test_plan(client):
    time.sleep(5)
    # testing models on nodes
    client.test()

def stop_now(client):
    # asking nodes to stop
    client.stop()


if __name__ == '__main__':
    logger.info("Central Hub initialized")

    global_round = 1

    for round in range(global_round):

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(train_plan, clients)    

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(aggregate, clients)   

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(test_plan, clients)    
        
        logger.info(f"Global round {round+1}/{global_round} completed")
        print("-------------------------------------------------")
    # all process excuted 

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(stop_now, clients)  
    logger.info("Done!")



#old aggregate function
# def aggregate():
#     for client in clients:
#         logger.info(f"Aggregating with Node: {client.address}...") 
#         checkpoint = client.gather()
#         for k in checkpoint.keys():
#             if k == "epoch":
#                 #epochs = checkpoint['epoch']
#                 logger.info(f"Best Epoch at Client: {checkpoint['epoch']}...") 
#             elif k == "weights":
#                 w = checkpoint['weights']
#                 logger.info("Copying weights...")
#                 w_loc.append(copy.deepcopy(w))
#                 logger.info("Aggregating weights...")
#                 w_glob = FedAvg(w_loc)
#             elif k == "metric":
#                 logger.info(f"Best Metric at Client: {checkpoint['metric']}..." )
#             else:
#                 logger.info('Server does not recognized the sent data')

#     cpt = {#'epoch': 1, # to be determined
#             'weights': w_glob#,
#             #'metric': 0 # to be aggregated
#             }
#     t.save(cpt, modelFile)
#     print(cpt.keys())
#     logger.info("aggregation completed")