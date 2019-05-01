import tensorflow as tf
import numpy as np

class WeightReader:
	def __init__(self, weight_file):
		self.offset = 4
		self.all_weights = np.fromfile(weight_file, dtype='float32')

	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset - size:self.offset]

	def reset(self):
		self.offset = 4