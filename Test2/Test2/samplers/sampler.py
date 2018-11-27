import numpy as np
from sklearn.model_selection import train_test_split

class Sampler():
  def __init__(self, batch_size=10, validation_size=0.1, seed=None):
    self.batch_size = batch_size
    self.validation_size = validation_size
    self.train_inputs = np.array([])
    self.train_outputs = np.array([])
    self.validation_inputs = np.array([])
    self.validation_outputs = np.array([])
    self.seed=seed

  def split_data(self, inputs, outputs):
    self.train_inputs, self.validation_inputs, self.train_outputs, self.validation_outputs = train_test_split(inputs, outputs, test_size=self.validation_size, random_state=self.seed)

  def get_next_batch(self):
    pass
  