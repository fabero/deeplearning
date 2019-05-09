from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Lambda, concatenate,Flatten,Softmax,Dense,Dropout,Activation,ZeroPadding2D


def identity_block(input_tensor, kernel_size, filters, stage, block,trainable=False):
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a',trainable=trainable)(x)
    x = Activation('relu',)(x)

    x = Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c',trainable=trainable)(x)

    x = concatenate([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),trainable=False):

    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a',trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b',trainable=trainable)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c',trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c',trainable=trainable)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1',trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1',trainable=trainable)(shortcut)

    x = concatenate([x, shortcut])
    x = Activation('relu')(x)
    return x



def get_feature_extractor_resnet50(input,trainable=False):
    bn_axis = 3
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad',trainable=trainable)(input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1',trainable=trainable)(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad',trainable=trainable)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2),trainable=trainable)(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',trainable=trainable)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',trainable=trainable)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',trainable=trainable)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',trainable=trainable)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',trainable=trainable)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',trainable=trainable)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',trainable=trainable)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',trainable=trainable)

    return x
