"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from Test2 import app

import database
import requests
import tensorflow as tf
import numpy as np
import freeze_graph
import time
from flask import jsonify

from Test2.converters.json_to_data import JSONToData
from Test2.data_preprocessing.generate_dataset import DatasetGenerator
from Test2.samplers.sampler import Sampler

api_key = 'ESFLMH9WRH6GLV7D'
base_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY'
trained_model_dir = './Test2/trained_models'

@app.route("/")
def home():
  return "This is home page."

@app.route('/model/train')
def train_model():
  return 'Success'

@app.route('/model/predict/<symbol>')
def predict(symbol):
  # Retrieve stock price data from API
  start_api = time.time()
  url = base_url + "&symbol={}&interval=1min&apikey={}".format(symbol, api_key)
  response = requests.get(url)
  end_api = time.time()
  start_predict = time.time()
  response_json = response.json()

  # Parse response into an array
  data = JSONToData().convertJson(response_json)

  # Generate input data for LSTM
  dataset_generator = DatasetGenerator()
  test_input = dataset_generator.generate_test_input(data[symbol])[None, :, 1:-1]

  # graph = tf.get_default_graph()
  graph = load_graph('{}/frozen_model.pb'.format(trained_model_dir))

  X = graph.get_tensor_by_name('X:0')
  outputs = graph.get_tensor_by_name('output:0')

  with tf.Session(graph=graph) as sess:
    saver = tf.train.import_meta_graph('{}/model.ckpt.meta'.format(trained_model_dir))
    saver.restore(sess, tf.train.latest_checkpoint(trained_model_dir))
    
    y_pred = sess.run(outputs, feed_dict={X: test_input})
  end_predict = time.time()
  temp_var = jsonify(predicted_price=np.array2string(y_pred[0, -1, 0]), api_latency=(end_api - start_api), prediction_latency=(end_predict - start_predict))
  return temp_var

def load_graph(frozen_graph_filename):
  # We load the protobuf file from the disk and parse it to retrieve the 
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we import the graph_def into a new Graph and returns it 
  with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="")
  return graph

@app.route("/data/<symbol>/full")
def retrieve_full_data_from_api(symbol):
  # Retrieve stock price data from API
  url = base_url + "&symbol={}&interval=1min&apikey={}&outputsize=full".format(symbol, api_key)
  response_json = requests.get(url).json()

  # Parse response into an array
  data = JSONToData().convertJson(response_json)

  # Generate data for LSTM
  dataset_generator = DatasetGenerator()
  dataset_generator.generate_dataset(data[symbol])
  
  # Split data into training and validation
  sampler = Sampler()
  sampler.split_data(dataset_generator.data_inputs, dataset_generator.data_outputs)

  # Construct LSTM to predict 10 steps from now. We are only interested in close price
  train_input = sampler.train_inputs[:, :, 1:-1]
  train_output = sampler.train_outputs[:, -2].reshape(-1, 1)
  test_input = sampler.validation_inputs[:, :, 1:-1]
  test_output = sampler.validation_outputs[:, -2].reshape(-1, 1)
  
  print(sampler.train_inputs)

  tf.reset_default_graph()

  num_periods = 100
  input_size = 4
  output_size = 1
  num_hidden = 100

  # Setup variables for graph
  X = tf.placeholder(tf.float32, [None, num_periods, input_size], name='X')
  y = tf.placeholder(tf.float32, [None, output_size], name='y')

  # Construct LSTM cell with specified num_hidden units
  basic_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)

  output, state = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

  stacked_rnn_output = tf.reshape(output, [-1, num_hidden])
  stacked_outputs = tf.layers.dense(stacked_rnn_output, output_size)
  outputs = tf.reshape(stacked_outputs, [-1, num_periods, output_size], name='output')

  # Squared Error Loss
  loss = tf.reduce_sum(tf.square(outputs[:,-1,:] - y))

  # Use Adam to perform gradient descent
  learning_rate = 0.002
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  training_op = optimizer.minimize(loss)
  
  saver = tf.train.Saver()

  # Train epochs (1000 is very slow (without GPU?))
  epochs = 5000
  prev_mse = np.Inf
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(epochs):
      sess.run(training_op, feed_dict={X: train_input, y: train_output})
      if ep % 10 == 0:
        mse = loss.eval(feed_dict={X: train_input, y: train_output})
        print('ep {}\tmse: {}'.format(ep, mse))
        if ((abs(prev_mse - mse) < 1e-4) or (prev_mse < mse)):
          break
        prev_mse = mse
  
    save_path = saver.save(sess, '{}/model.ckpt'.format(trained_model_dir))
    print("Model saved in path: %s" % save_path)
  
  # Freeze graph for prediction
  freeze_graph.freeze_graph(trained_model_dir, 'X,y,output')

  return 'Success'

@app.route('/data/<symbol>')
def retrieve_compact_data_from_api(symbol):
  url = base_url + "&symbol={}&interval=1min&apikey={}".format(symbol, api_key)
  r = requests.get(url)
  print(r.text)
  return r.text
  
db = database.create_db()