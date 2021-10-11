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
import torch as t
from coordinator import FedAvg
import shutil
from datetime import datetime
import json
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

FL_CLIENT_ENDPOINTS = json.loads(os.environ.get('FL_CLIENT_ENDPOINTS'))
ENVIRONMENT = os.environ.get('ENVIRONMENT')
MODEL_ID = os.environ.get('MODEL_ID')

modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"

modelFile = os.path.join(modelpath, modelName)
w_loc = []
clients = [Client(address) for address in FL_CLIENT_ENDPOINTS]

def model_spread_plan(client):
    try:
        if(client.status() == "alive"):
            # spreading model to nodes
            client.bootstrap()
    except:
        logger.info(f"client {client.address} is dead...")

def train_plan(client):
    try:
        if(client.status() == "alive"):
            # initializing training on nodes
            client.train(epochs='1')
    except:
        logger.info(f"client {client.address} is dead...")
    
def aggregate(client):
    try:
        if(client.status() == "alive"):
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
    except:
        logger.info(f"client {client.address} is dead...")

def test_plan(client):
    try:
        if(client.status() == "alive"):
            # testing models on nodes
            client.test()
    except:
        logger.info(f"client {client.address} is dead...")
    
def stop_now(client):
    try:
        if(client.status() == "alive"):
            # asking nodes to stop
            client.stop()
    except:
        logger.info(f"client {client.address} is dead...")

def upload_results_in_s3_bucket(source_path: str, bucket_name: str = 'flip-uploaded-federated-data-bucket'):
    bucket_name += '-' + ENVIRONMENT
    
    logger.info('Zipping the final model and the test reports...')
    zip_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(cwd, "save", zip_name)
    shutil.make_archive(zip_path, 'zip', source_path)

    logger.info(f'Uploading zip file {zip_path} to S3 bucket {bucket_name} in folder {MODEL_ID}...')
    bucket_zip_path = MODEL_ID + '/' + zip_name
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(zip_path + '.zip', bucket_name, bucket_zip_path + '.zip')
    except ClientError as e:
        logger.error(e)


if __name__ == '__main__':
    logger.info("Central Hub initialized")

    global_round = 1

    for round in range(global_round):
        if (round==0):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                result = executor.map(model_spread_plan, clients)    

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(train_plan, clients)    

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(aggregate, clients)   

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(test_plan, clients)    
        
        logger.info(f"Global round {round+1}/{global_round} completed")
        print("-------------------------------------------------")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(stop_now, clients)  
    
    # all processes are excuted 
    logger.info(f"Model Training is completed across all sites and current global model is available at following location...{modelFile}")

    upload_results_in_s3_bucket(modelpath)

    logger.info("Centra Hub FL Server terminated")