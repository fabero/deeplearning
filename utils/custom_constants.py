

IMG_H = 200 #height of the image
IMG_W= 200 #width of the image

LABELS = ['cat', 'car', 'fruit', 'dog', 'person', 'flower', 'motorbike', 'airplane'] #list of labels
LABEL_IDS = {'cat':0,'car':1, 'fruit':2, 'dog':3, 'person':4, 'flower':5, 'motorbike':6, 'airplane':7}
CLASS = len(LABELS)#Number of classes
EPOCHS = 50
BATCH_SIZE = 20

weights_path_vgg19 = './transfer_learning/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

weights_path_resnet50='./transfer_learning/resnet50weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
