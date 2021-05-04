# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, request, jsonify
from utils import *

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

num_elements = 16000
# Setting the maximum value in the randomly generated list.
max_num = 1000

# Instantiate a manager object to manage the data chunks, num of chunks = (nprocs - 1)
data_manager_master = DataManager(num_elements, 16 , max_num)
merger = Merger([])

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def get_id():
	return {'Message':'Hello'}
	
	
@app.route('/get_data')
# ‘/’ URL is bound with hello_world() function.
def getData():
	arr_to_send, chunk_num = data_manager_master.get_next_chunk()
	return {'Data':arr_to_send.tolist(), 'ID': chunk_num}


@app.route('/post_data', methods=['POST'])
# ‘/’ URL is bound with hello_world() function.
def postData():
	data = request.get_json()
	message = 'Received Data from ' + str(data['ID'])
	print(message)
	return {'Data':'Thanks..'}
# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run()

