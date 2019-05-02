import numpy as np

IMG_H = 64 #height of the image
IMG_W= 64 #width of the image
GRID_H = 13
GRID_W = 13#size of grid
LABELS = ['0','1','2','3','4','5','6','7','8','9'] #list of labels
CLASS = len(LABELS) #number of classes
#TODO: Update this understanding if you find it different
CLASS_WEIGHTS = np.ones(CLASS, dtype='float32') #weight for each class, assuming % of every class in data base is equal


ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

# TODO: See what is this
TRUE_BOX_BUFFER  = 50
COORD_SCALE = 1.0
NO_OBJECT_SCALE = 1.0
OBJECT_SCALE = 5.0
CLASS_SCALE = 1.0
WARM_UP_BATCHES = 0
BATCH_SIZE = 16

BOX= 5