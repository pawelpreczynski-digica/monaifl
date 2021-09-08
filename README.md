# fl-architecture
This repository contains the basic example of monai code considering substra's opener and algo classes.

Reference notebook

https://github.com/Project-MONAI/tutorials/blob/master/2d_classification/mednist_tutorial.ipynb

<<<<<<< HEAD
How to Install:
=======
Installation:
>>>>>>> 35a206fbce0103377b27a0e6e9c64b20c09ddb97
 - create conda environment with python 3.8
 - install monai
 - install grpcio
 - install grpc with python compiler
    - python -m pip install grpcio
    - python -m pip install grpcio-tools
 - install your system compatible pytorch version
 - install pandas
 - install numpy

<<<<<<< HEAD
Running Server:
1. Open a new terminal
2. activate conda environment
3. $ cd monaifl
4. $monaifl/python aggregator/coordinator/src/server.py

Running Client:
1. Open a new terminal
2. activate conda environment
3. cd monaifl 
4. python trainer/substra/testmonai.py
=======
How to run:

Run the server:
 - navigate to source file (currently it is not dockerized)
   - cd monaifl/aggregator/coordinator/src
 - run the server
   - python server.py  

Mount the dataset:
 This repository contains the demo dataset. If you are cloning this repository, you do not need to mount any dataset. Otherwise, you must execute the cell 5 from the reference notebook, extract the dataset at your preferred location and set the path in testmonai.py 

Run the client:
 - navigate to source file (the entry point from the client side)
  - cd monaifl/trainer/substra
 - run the example
  - python testmonai.py

CAUTION: The server side code does not require any GPU installation but for the client side you must have a CUDA-enabled device to initiate the training. 

Please feel free to contact me at habibcomsats@gmail.com or mhrehman@ieee.org for any query, feedback, suggestion, or discussion.
>>>>>>> 35a206fbce0103377b27a0e6e9c64b20c09ddb97
