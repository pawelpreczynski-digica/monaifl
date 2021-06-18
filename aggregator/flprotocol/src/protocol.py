# This file will be the entry point for protocol container. 
# It will enable various training protocols and aggregation stretgies
# It will monitor the global training performance
# it will enable the selection/deseletion of training nodes
# it will enable faireness, data distribution, and all other relevant FL properties for a decentralized and trustworthy FL system

import os
import subprocess
import py_compile
from subprocess import Popen

from pathlib import Path
home = str(Path.home())

print(home)

serverPath = os.path.join(home, "monaifl", "aggregator", "coordinator","src")
serverFile = "server.py"
server = os.path.join(serverPath, serverFile)

clientPath= os.path.join(home, "monaifl", "trainer", "reporter","src")
clientFile = "client.py"
client = os.path.join(clientPath, clientFile)

pipelinePath= os.path.join(home, "monaifl", "trainer", "MONAI","pipeline")
#codeFile = "mednist.py"

pipelinePath= os.path.join(home, "monaifl", "trainer", "MNIST")
codeFile = "testMNIST.py"

pipeline = os.path.join(pipelinePath, codeFile)

logpathglobal = os.path.join(home, "monaifl", "save","logs","client")
logNameglobal = 'globalmnistlog.txt'
logFileGlobal = os.path.join(logpathglobal, logNameglobal)


def flprotocol():
    if (os.path.exists(logFileGlobal)):
        file = open(logFileGlobal,"w")
        file.truncate(0)
        file.close()
    p = Popen(["python", server])
    for i in range(5):
        print("Initial Global Model Transferred!")
        p = Popen(["python", pipeline])
        p.wait()        
        p = Popen(["python", client])

flprotocol()
