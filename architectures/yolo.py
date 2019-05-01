from keras.models import Model,Sequential
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, concatenate, Reshape
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop

from utils.utils import WeightReader
from utils.loss import custom_loss
from utils.constants import *
'''
This is YOLOV2
'''


input_image= Input(shape=(IMG_H, IMG_W, 3)) #structure of input
true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

class YoloNetwork():
	def __init__(self,config_pickle=False):
		#Initialise constants over here.

		if not config_pickle:
			pass
		else:
			pass

	def set_constants_from_config(self,config):
		'''
		TODO: can define a wrapper to check whether config file is of proper format
		Set class constants from config file provided,
		:param config: Config file containing constant values
		:return:
		'''
		pass


	# def create_input_structure(self):
	# 	'''
	# 	instantiate input tensors
	# 	:return:
	# 	'''
	# 	#TODO: Make changes in case of batch
	# 	self.input_image= Input(shape=(IMG_H, IMG_W, 3)) #structure of input
	#
	# 	self.true_boxes = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

	@staticmethod
	def __space_to_depth_x2(x):
		return tf.space_to_depth(x, block_size=2)

	def create_yolo_layers(self,input_image,true_boxes):
		'''
		Create all the layers of the yolo
		:param input_image:
		:param true_boxes
		:return:
		'''

		'''
		#Layer1: 3X3 kernel with 32 filters 
		'''
		x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
		x = BatchNormalization(name='norm_1')(x)
		x = LeakyReLU(alpha=0.1)(x)
		#x = MaxPooling2D(pool_size=(2, 2))(x) #Size WxH

		# Layer 2
		x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
		x = BatchNormalization(name='norm_2')(x)
		x = LeakyReLU(alpha=0.1)(x)
		#x = MaxPooling2D(pool_size=(2, 2))(x) #Size WxH

		# Layer 3
		x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
		x = BatchNormalization(name='norm_3')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 4
		x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
		x = BatchNormalization(name='norm_4')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 5
		x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
		x = BatchNormalization(name='norm_5')(x)
		x = LeakyReLU(alpha=0.1)(x)
		#x = MaxPooling2D(pool_size=(2, 2))(x) #Size WxH

		# Layer 6
		x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
		x = BatchNormalization(name='norm_6')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 7
		x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
		x = BatchNormalization(name='norm_7')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 8
		x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
		x = BatchNormalization(name='norm_8')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x) #Size W/2 X H/2 = 32x32

		# Layer 9
		x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
		x = BatchNormalization(name='norm_9')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 10
		x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
		x = BatchNormalization(name='norm_10')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 11
		x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
		x = BatchNormalization(name='norm_11')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 12
		x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
		x = BatchNormalization(name='norm_12')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 13
		x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13_a', use_bias=False)(x)

		# Decrease size from 32x32 to 26x26 using conv layer
		x = Conv2D(512, (7, 7), strides=(1, 1), padding='valid', name='conv_13', use_bias=False)(x)

		x = BatchNormalization(name='norm_13')(x)
		x = LeakyReLU(alpha=0.1)(x)





		skip_connection = x #String output till here in skip connection

		x = MaxPooling2D(pool_size=(2, 2))(x)

		# Layer 14
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
		x = BatchNormalization(name='norm_14')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 15
		x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
		x = BatchNormalization(name='norm_15')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 16
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
		x = BatchNormalization(name='norm_16')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 17
		x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
		x = BatchNormalization(name='norm_17')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 18
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
		x = BatchNormalization(name='norm_18')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 19
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
		x = BatchNormalization(name='norm_19')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 20
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
		x = BatchNormalization(name='norm_20')(x)
		x = LeakyReLU(alpha=0.1)(x)

		# Layer 21
		skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
			skip_connection)
		skip_connection = BatchNormalization(name='norm_21')(skip_connection)
		skip_connection = LeakyReLU(alpha=0.1)(skip_connection)

		skip_connection = Lambda(YoloNetwork.__space_to_depth_x2)(skip_connection)

		x = concatenate([skip_connection, x])

		# Layer 22
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
		x = BatchNormalization(name='norm_22')(x)
		x = LeakyReLU(alpha=0.1)(x)
		print(x.shape)
		# Layer 23
		x = Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
		print(x.shape)




		output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
		print(output.shape)
		#output=x

		#TODO: What are we doing here
		output = Lambda(lambda args: args[0])([output, true_boxes])
		print(output.shape)

		self.model = Model([input_image, true_boxes], output)

		self.nb_conv = 23 #No. of convolution layers


	def load_weights_from_file(self,weight_path):
		self.model.load_weights(weight_path)

	def save_weights_to_file(self,path_to_save):
		self.model.save_weights(path_to_save)

	def reset_weights(self):
		session = K.get_session()
		for layer in self.model.layers:
			if hasattr(layer, 'kernel_initializer'):
				layer.kernel.initializer.run(session=session)

	def initialise_weights(self):
		self.reset_weights()


	# def load_pretrained_weights(self,wt_path):
	# 	'''
	# 	:param wt_path:
	# 	:return:
	# 	'''
	#
	# 	weight_reader = WeightReader(wt_path)
	#
	# 	for i in range(1, self.nb_conv + 1):
	# 		conv_layer = self.model.get_layer('conv_' + str(i))
	#
	# 		if i < self.nb_conv:
	# 			norm_layer = self.model.get_layer('norm_' + str(i))
	#
	# 			size = np.prod(norm_layer.get_weights()[0].shape)
	#
	# 			beta = weight_reader.read_bytes(size)
	# 			gamma = weight_reader.read_bytes(size)
	# 			mean = weight_reader.read_bytes(size)
	# 			var = weight_reader.read_bytes(size)
	#
	# 			weights = norm_layer.set_weights([gamma, beta, mean, var])
	#
	# 		if len(conv_layer.get_weights()) > 1:
	# 			bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
	# 			kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
	# 			kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
	# 			kernel = kernel.transpose([2, 3, 1, 0])
	# 			conv_layer.set_weights([kernel, bias])
	# 		else:
	# 			kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
	# 			kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
	# 			kernel = kernel.transpose([2, 3, 1, 0])
	# 			conv_layer.set_weights([kernel])


	#TODO: Look into theory behind this
	def randomize_weights_last_layer(self):
		layer = self.model.layers[-4]  # the last convolutional layer
		weights = layer.get_weights()

		new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
		new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)

		#TODO: check if it actually gets changed
		layer.set_weights([new_kernel, new_bias])



	@staticmethod
	def custom_loss(y_true, y_pred):
		return custom_loss(y_true,y_pred,
			GRID_W,
			GRID_H,
			BATCH_SIZE,
			ANCHORS,
			BOX,
			true_boxes,
			COORD_SCALE,
			NO_OBJECT_SCALE,
			OBJECT_SCALE,
			CLASS_SCALE,
			CLASS_WEIGHTS,
			WARM_UP_BATCHES)



	def set_up(self):
		#self.create_input_structure()
		self.create_yolo_layers(input_image,true_boxes)

		self.model.summary()
		self.initialise_weights()
		self.randomize_weights_last_layer()



		optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		# optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
		# optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

		self.model.compile(loss= YoloNetwork.custom_loss,optimizer=optimizer)

