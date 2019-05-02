
import cv2 as cv
from keras.utils import Sequence,to_categorical
from utils.custom_constants import *
import numpy as np
import os

class DataGenerator(Sequence):

	def __init__(self,data_folder,shuffle=True):
		self.data_folder = data_folder
		self.all_files=[]
		self.shuffle = shuffle

		self.set_all_image_info()


		self.on_epoch_end()


		self.batch_size = BATCH_SIZE
		self.n_classes = CLASS
		self.dim = (IMG_H, IMG_W, 3)

	def set_all_image_info(self):
		'''
		This function will parse all files from the data folder and store category name and id
		:return:
		'''
		self.num_of_files=0
		for dirname, dirnames, filenames in os.walk(self.data_folder):
			for subdirname in dirnames:
				pass


			for filename in filenames:
				if not filename.startswith('.'):
					self.num_of_files=self.num_of_files+1
					category = dirname.split('/')[-1]
					id=LABEL_IDS[category]
					self.all_files.append((dirname,category,filename,id))


	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.all_files))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.all_files) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

		# Find list of image_info
		list_image_info_temp = [self.all_files[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_image_info_temp)

		return X, y


	def load_image(self,image_path):
		image = cv.imread(image_path,cv.IMREAD_UNCHANGED)
		new_dim = (IMG_H, IMG_W) #as all images are of different size so we need to resize
		image = cv.resize(image,new_dim)
		return image

	def __data_generation(self, list_image_info_temp):
		'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size, *self.dim))
		y = np.empty((self.batch_size), dtype=int)

		# Generate data
		for i, image_info in enumerate(list_image_info_temp):
			# Store sample
			#print('***',image_info)
			X[i,] = self.load_image(image_info[0] + '/' + image_info[2])

			# Store class
			y[i] = image_info[3] #Last element of the list is id

		return X, to_categorical(y, num_classes=self.n_classes)