from pathlib import Path
cwd = str(Path.cwd())

import sys
sys.path.append('.')
import os
from hub import Client, Stage
import logging
import concurrent.futures
import shutil
from datetime import datetime
import json
import boto3

FL_CLIENT_ENDPOINTS = json.loads(os.environ.get('FL_CLIENT_ENDPOINTS'))
ENVIRONMENT = os.environ.get('ENVIRONMENT')
MODEL_ID = os.environ.get('MODEL_ID')

modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"
modelFile = os.path.join(modelpath, modelName)

federated_process_logger = logging.getLogger('federated_process')
syslog = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s]-[%(model_id)s]-[%(status)s]-[%(trust_name)s]-%(message)s')
syslog.setFormatter(formatter)
federated_process_logger.setLevel(logging.INFO)
federated_process_logger.addHandler(syslog)

node_status_logger = logging.getLogger('node_status')
syslog = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s]-[%(model_id)s]-[%(trust_name)s]-%(message)s')
syslog.setFormatter(formatter)
node_status_logger.setLevel(logging.INFO)
node_status_logger.addHandler(syslog)

main_logger = logging.getLogger('main')
syslog = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s]-[%(model_id)s]-[%(status)s]-%(message)s')
syslog.setFormatter(formatter)
main_logger.setLevel(logging.INFO)
main_logger.addHandler(syslog)
main_logger = logging.LoggerAdapter(main_logger, extra={'model_id': MODEL_ID})

logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def model_spread_plan(client):
    try:
        if(client.status() == "alive"):
            # spreading model to nodes
            client.bootstrap()
    except:
        logger.info(f"client {client.name} is dead...")

def train_plan(client):
    try:
        if(client.status() == "alive"):
            # initializing training on nodes
            client.train(epochs='1')
    except:
        logger.info(f"client {client.name} is dead...")
    
def aggregate(client):
    try:
        if(client.status() == "alive"):
            # aggregating models from nodes
            client.aggregate()
    except Exception as ex:
        print(ex)
        logger.info(f"client {client.name} is dead...")

def test_plan(client):
    try:
        if(client.status() == "alive"):
            # testing models on nodes
            client.test()
    except:
        logger.info(f"client {client.name} is dead...")
    
def stop_now(client):
    try:
        if(client.status() == "alive"):
            # asking nodes to stop
            client.stop()
    except:
        logger.info(f"client {client.name} is dead...")

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
    except Exception as e:
        logger.error(e)

    logger.info('Upload completed!')


if __name__ == '__main__':
    logger.info("hub started")
    clients = [Client(fl_client['flclientendpoint'], fl_client['name']) for fl_client in FL_CLIENT_ENDPOINTS]

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

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(stop_now, clients)  
    
    # all processes are excuted 
    logger.info(f"Model Training is completed across all sites and current global model is available at {modelFile}")

    upload_results_in_s3_bucket(modelpath)

    logger.info("Centra Hub FL Server terminated")