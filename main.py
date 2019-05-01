
from architectures.yolo import YoloNetwork
from architectures.customArchitecture import CustomNetwork

from utils.create_data import CreateData

if __name__ =="__main__":
	# yolo =YoloNetwork()
	# yolo.set_up()

	arch = CustomNetwork(training_settings_name='initial')
	arch.set_up()

	# cd = CreateData('../data/natural_images/')
	# cd.create_train_test_validation_data()
