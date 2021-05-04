# importing the requests library
import requests
from utils import *

# api-endpoint
URL = "http://127.0.0.1:5000/"
sort_machine = Sorting()



# sending get request and saving the response as response object
r = requests.get(url = f'{URL}get_data')

# extracting data in json format
data = r.json()
sorted_array = sort_machine.sort_arr(data['Data'])
node_id = data['ID']
name = 'My id is ' + str(node_id) 
print(name)
print(sorted_array)

# sending get request and saving the response as response object
r = requests.post(url = f'{URL}post_data', json = {'Data':sorted_array, 'ID': node_id})

