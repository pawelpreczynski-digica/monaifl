import sys
sys.path.append('.')

from substraclient import Client
import time
import logging
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

if __name__ == '__main__':
    client = Client("localhost:50051")
    logger.info("Central Hub initialized")

    client.bootstrap()
    
    client.train(epochs='1')

    trained = client.status()
    while (trained != "Training completed"):
        time.sleep(5)
        trained = client.status()
        
    checkpoint = client.gather()  
    
    client.test()

    client.stop()

