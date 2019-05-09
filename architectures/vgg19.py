from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, concatenate,Flatten,Softmax,Dense,Dropout


def get_feature_extractor_vgg19(input,trainable=False):
	x = Conv2D(64, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block1_conv1',trainable=trainable)(input)
	x = Conv2D(64, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block1_conv2',trainable=trainable)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',trainable=trainable)(x)

	# Block 2
	x = Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv1',trainable=trainable)(x)
	x = Conv2D(128, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block2_conv2',trainable=trainable)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',trainable=trainable)(x)

	# Block 3
	x = Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv1',trainable=trainable)(x)
	x = Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv2',trainable=trainable)(x)
	x = Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv3',trainable=trainable)(x)
	x = Conv2D(256, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block3_conv4',trainable=trainable)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',trainable=trainable)(x)

	# Block 4
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv1',trainable=trainable)(x)
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv2',trainable=trainable)(x)
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv3',trainable=trainable)(x)
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block4_conv4',trainable=trainable)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',trainable=trainable)(x)

	# Block 5
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv1',trainable=trainable)(x)
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv2',trainable=trainable)(x)
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv3',trainable=trainable)(x)
	x = Conv2D(512, (3, 3),
					  activation='relu',
					  padding='same',
					  name='block5_conv4',trainable=trainable)(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',trainable=trainable)(x)

	return x