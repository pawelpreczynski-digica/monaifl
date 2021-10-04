# import asyncio

# async def fetch_data():
#     print('start fetching')
#     await asyncio.sleep(2)
#     print('done fetching')
#     return {'data':1}

# async def print_numbers():
#     for i in range(10):
#         print(i)
#         await asyncio.sleep(0.5)


# async def main():
#     task1 = asyncio.create_task(fetch_data())
#     task2 = asyncio.create_task(print_numbers())

#     value = await task1
#     print(value)
#     await task2


# asyncio.run(main())
    
# import sys
# sys.path.append('.')

# from substraclient import Client
# import time
# import logging
# import threading
# import asyncio
# import time
# logging.basicConfig(format='%(asctime)s - %(message)s')
# logger = logging.getLogger()
# logger.setLevel(logging.NOTSET)

# async def start_training(client):
#     """
#     function to initiate traing at the client
#     """
#     await client.train(epochs='1')
  
# async def check_status(client):
#     """
#     function to check the traing status.
#     """
#     await client.status()

# async def plan():
#     client = Client("localhost:50051")
#     logger.info("Central Hub initialized")

#     client.bootstrap()
    
#     # creating tasks
#     task1  = asyncio.create_task(start_training(client))
#     task2  = asyncio.create_task(check_status(client))

#     await task2 
#     await task1
       
#     client.gather()  
    
#     client.test()

#     client.stop()

#     # plan completely executed
#     print("Done!")


    

# if __name__ == '__main__':
# #    asyncio.run(plan())
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(plan())



#  # creating thread
#     # t1 = threading.Thread(target=start_training, args=(client,))
#     # t2 = threading.Thread(target=check_status, args=(client,))
  
#     # # starting thread 1
#     # t1.start()
#     # # starting thread 2
#     # t2.start()
  
#     # # wait until thread 1 is completely executed
#     # t1.join()
#     # # wait until thread 2 is completely executed
#     # t2.join()
     