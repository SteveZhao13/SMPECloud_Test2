import numpy as np

class DatasetGenerator():
  def __init__(self, input_size=100, input_output_gap=10):
    self.input_size = input_size
    self.input_output_gap = input_output_gap
    self.data_inputs = np.array([])
    self.data_outputs = np.array([])

  def generate_test_input(self, data):
    return np.array(data)
  
  def generate_dataset(self, data):
    '''
    Generate the data set using data, based on
    input_size: The amount of x to predict y
    input_output_gap: The gap between the latest x and y
    '''
    num_data = len(data) - self.input_size - self.input_output_gap + 1
    self.data_outputs = np.array(data[self.input_size + self.input_output_gap - 1:])
    self.data_inputs = np.array([data[i:i + self.input_size] for i in range(num_data)])