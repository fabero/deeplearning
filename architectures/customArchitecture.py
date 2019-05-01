from keras.models import Model,Sequential
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, concatenate, Reshape,Flatten,Softmax,Dense,Dropout
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.optimizers import SGD, Adam, RMSprop

from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
from utils.data_generator import DataGenerator

from utils.custom_constants import *
import os
'''
This is Custom
'''


input_image= Input(shape=(IMG_H, IMG_W, 3)) #structure of input

class CustomNetwork():
	def __init__(self,config_pickle=False,training_settings_name='initial'):
		self.training_settings_name = training_settings_name
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

	def create_feature_extractor_layers(self,input):
		'''
		:param true_boxes
		:return:
		'''

		'''
		#Layer1: 3X3 kernel with 32 filters 
		'''
		x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input)
		x = BatchNormalization(name='norm_1')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x) #Size WxH

		# Layer 2
		x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
		x = BatchNormalization(name='norm_2')(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x) #Size WxH

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
		x = MaxPooling2D(pool_size=(2, 2))(x) #Size WxH

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

		#print(x.shape)

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
		x = Conv2D(512, (3, 3), strides=(1, 1), padding='valid', name='conv_13', use_bias=False)(x)

		# Decrease size from 32x32 to 26x26 using conv layer
		#x = Conv2D(512, (7, 7), strides=(1, 1), padding='valid', name='conv_13', use_bias=False)(x)

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

		skip_connection = Lambda(CustomNetwork.__space_to_depth_x2)(skip_connection)

		x = concatenate([skip_connection, x])

		# Layer 22
		x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
		x = BatchNormalization(name='norm_22')(x)
		x = LeakyReLU(alpha=0.1)(x)


		# Layer 23
		x = Conv2D(60, (3, 3), strides=(1, 1), padding='same', name='conv_23', use_bias=False)(x)
		x = BatchNormalization(name='norm_23')(x)
		x = LeakyReLU(alpha=0.1)(x)

		return x




	def create_classifier_layers(self,input):
		# Layer 24, Fully connected
		x = Flatten()(input)
		# print(x.shape)
		x = Dense(units=100)(x)
		# print(x.shape)
		x = LeakyReLU(alpha=0.1)(x)
		# print(x.shape)

		x = Dropout(0.3)(x)
		# print(x.shape)
		x = Dense(units=CLASS)(x)
		# print(x.shape)

		output = Softmax()(x)
		# print(x.shape)

		return output




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


	def get_early_stoping_callback(self):
		early_stop = EarlyStopping(monitor='val_loss',
								   min_delta=0.001,
								   patience=3,
								   mode='min',
								   verbose=1)
		return early_stop

	def get_tensorboard_log_callback(self):
		tb_counter = len([log for log in os.listdir('./logs/') if 'custom_' in log]) + 1
		tensorboard = TensorBoard(log_dir= './logs/'+ 'custom_' + '_' + str(tb_counter),
								  histogram_freq=0,
								  write_graph=True,
								  write_images=False)
		return tensorboard


	def get_checkpoint_callback(self):
		checkpoint = ModelCheckpoint('./models'+self.training_settings_name+'/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
									 monitor='val_loss',
									 verbose=1,
									 save_best_only=True,
									 save_weights_only=True,
									 mode='min',
									 period=1)
		return checkpoint




	def set_up(self):
		#self.create_input_structure()
		extracted_features=self.create_feature_extractor_layers(input_image)
		classifier_output=self.create_classifier_layers(extracted_features)

		self.model = Model([input_image], classifier_output)

		#self.model.summary()

		self.initialise_weights()



		optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		# optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
		# optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

		self.model.compile(loss= 'categorical_crossentropy',optimizer=optimizer)

		early_stop = self.get_early_stoping_callback()
		checkpoint = self.get_checkpoint_callback()
		tensorboard = self.get_tensorboard_log_callback()

		training_generator = DataGenerator('../data/train/')
		valid_generator = DataGenerator('../data/val/')

		self.model.fit_generator(generator=training_generator,
							steps_per_epoch=len(training_generator),
							epochs=EPOCHS,
							verbose=1,
							validation_data=valid_generator,
							validation_steps=len(valid_generator),
							callbacks=[early_stop, checkpoint, tensorboard],
							max_queue_size=3)

