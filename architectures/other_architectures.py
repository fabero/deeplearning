import numpy as np
import tensorflow as tf


from tensorflow.python.keras.layers import Dense, Flatten, BatchNormalization,LeakyReLU,ReLU,PReLU,ELU,Dropout,Softmax
from tensorflow.python.keras.optimizers import Adam,RMSprop,SGD

from tensorflow.python.keras.models import Sequential

from keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint
from utils.data_generator import DataGenerator

from utils.custom_constants import *
import os


from tensorflow.python.keras.applications import ResNet50,VGG19,InceptionV3
'''
This is Custom
'''

input_shape=(IMG_H, IMG_W, 3)

class OtherNetwork():
	def __init__(self,config_pickle=False,training_settings_name='initial'):
		self.training_settings_name = training_settings_name
		#Initialise constants over here.

		self.check_and_set_required_dirs() #It will make and set logs dir and models dir

		if not config_pickle:
			pass
		else:
			pass


	def get_activation(self):
		activation_type = self.activation_type
		if activation_type ==1:
			activation = LeakyReLU(alpha=0.1)
		elif activation_type == 2:
			activation = ReLU()
		elif activation_type == 3:
			activation = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
		elif activation_type == 4:
			activation=ELU(alpha=1.0)
		else:
			raise Exception('Not a valid activation type')

		return activation

	def set_activation_type(self,activation_type=1):
		'''
		:param activation_type: 1-> LeakyRelu, 2-> Relu, 3-> PreRelu,4-> ELU
		:return:
		'''
		self.activation_type = activation_type


	def check_and_set_required_dirs(self):
		self.logs_dir = './logs/'+self.training_settings_name+'/'
		self.models_dir= './models/'+ self.training_settings_name+'/'

		if not os.path.exists(os.path.dirname(self.models_dir)):
			try:
				os.makedirs(os.path.dirname(self.models_dir))
			except Exception as ex:
				print(ex)
				pass

		if not os.path.exists(os.path.dirname(self.logs_dir)):
			try:
				os.makedirs(os.path.dirname(self.logs_dir))
			except Exception as ex:
				print(ex)
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





	def create_classifier_layers(self,add_dropout=True):
		self.model.add(Flatten())
		self.model.add(Dense(units=100))
		self.model.add(self.get_activation())
		if add_dropout:
			self.model.add(Dropout(0.3))

		self.model.add(Dense(units=CLASS))
		self.model.add(Softmax())
		return

	# def create_classifier_layer_2(self):
	# 	self.model.add(BatchNormalization())
	# 	self.model.add(Dense(2048, activation='relu'))
	# 	self.model.add(BatchNormalization())
	# 	self.model.add(Dense(1024, activation='relu'))
	# 	self.model.add(BatchNormalization())
	# 	self.model.add(Dense(num_classes, activation='softmax'))




	def load_weights_from_file(self,weight_path):
		self.model.load_weights(weight_path)

	def save_weights_to_file(self,path_to_save):
		self.model.save_weights(path_to_save)


	def get_early_stoping_callback(self):
		early_stop = EarlyStopping(monitor='val_loss',
								   min_delta=0.001,
								   patience=3,
								   mode='min',
								   verbose=1)
		return early_stop

	def get_tensorboard_log_callback(self):
		tb_counter = len([log for log in os.listdir(self.logs_dir) if 'custom_' in log]) + 1
		tensorboard = TensorBoard(log_dir= self.logs_dir+ 'custom_' + '_' + str(tb_counter),
								  histogram_freq=0,
								  write_graph=True,
								  write_images=False)
		return tensorboard


	def get_checkpoint_callback(self):
		checkpoint = ModelCheckpoint(self.models_dir+'weights.{epoch:02d}-{val_loss:.2f}.hdf5',
									 monitor='val_loss',
									 verbose=1,
									 save_best_only=True,
									 save_weights_only=True,
									 mode='min',
									 period=1)
		return checkpoint



	def train(self,
			  train_set_folder,
			  val_set_folder,
			  initial_weights_path=None,
			  initial_epoch=0,
			  optimizer=1,
			  loss=1,
			  activation=1,
			  add_dropout=True,
			  feature_extractor=2):
		'''

		:param train_set_folder:
		:param val_set_folder:
		:param initial_weights_path:
		:param initial_epoch:
		:param optimizer: 1-> Adam, 2-> SGD, 3-> RMSprop
		:param loss: 1-> categorical_crossentropy
		:param network:1 -> means feature_extractor is custom
		:return:
		'''

		self.set_activation_type(activation_type=activation)
		self.set_up(add_dropout=add_dropout,feature_extractor=feature_extractor) #SET up model



		self.model.summary()

		if optimizer==1:
			optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		elif optimizer ==2:
			optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
		elif optimizer==3:
			optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)
		else:
			raise Exception('Not a valid number for optimizer')

		self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

		#return
		early_stop = self.get_early_stoping_callback()
		checkpoint = self.get_checkpoint_callback()
		tensorboard = self.get_tensorboard_log_callback()



		training_generator = DataGenerator(train_set_folder)
		valid_generator = DataGenerator(val_set_folder)

		self.model.fit_generator(generator=training_generator,
								 steps_per_epoch=len(training_generator),
								 epochs=EPOCHS,
								 verbose=1,
								 validation_data=valid_generator,
								 validation_steps=len(valid_generator),
								 callbacks=[early_stop, checkpoint, tensorboard],
								 max_queue_size=3
								 #use_multiprocessing=True,
								 #initial_epoch=initial_epoch
		)

	def test(self,weights_path):
		if not weights_path:
			raise Exception('Weights not given')

		self.set_up()

		self.load_weights_from_file(weights_path)


	def set_up(self,add_dropout=True,feature_extractor=2):
		#self.create_input_structure()

		if feature_extractor == 2:#ResNet50
			weights_path='./transfer_learning/resnet50weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
			base_model=ResNet50(include_top=False,input_shape=input_shape,weights=weights_path)



			self.model = Sequential()

			self.model.add(base_model)

			self.create_classifier_layers(add_dropout=add_dropout)



		elif feature_extractor ==3:
			weights_path = './transfer_learning/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
			base_model = VGG19(include_top=False, input_shape=input_shape,
										  weights=weights_path)

			self.model = Sequential()

			self.model.add(base_model)

			self.create_classifier_layers(add_dropout=add_dropout)

		else:
			raise Exception('Feature extractor not defined')



		#self.model = Model([input_image], classifier_output)




