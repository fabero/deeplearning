
from architectures.yolo import YoloNetwork
from architectures.customArchitecture import CustomNetwork

from utils.create_data import CreateData

import argparse



if __name__ =="__main__":
	parser = argparse.ArgumentParser(description='Argument Parser')

	parser.add_argument('--training_settings_name', type=str, default='initial')
	parser.add_argument('--train_set_folder', type=str,default='../data/train/')
	parser.add_argument('--val_set_folder', type=str, default='../data/val/')

	parser.add_argument('--initial_weights_path', type=str, default=None)
	parser.add_argument('--initial_epoch', type=int, default=0)
	parser.add_argument('--optimizer', type=int, default=1)
	parser.add_argument('--loss', type=int, default=1)
	parser.add_argument('--activation', type=int, default=1)
	parser.add_argument('--add_dropout', type=int, default=1)
	args=parser.parse_args()


	arch = CustomNetwork(training_settings_name=args.training_settings_name)
	arch.train(args.train_set_folder,
			   args.val_set_folder,
			   initial_weights_path=args.initial_weights_path ,
			   initial_epoch=args.initial_epoch,
			   optimizer=args.optimizer,
			   loss=args.loss,
			   activation=args.activation,
			   add_dropout= True if args.add_dropout else False
			   )

	# cd = CreateData('../data/natural_images/')
	# cd.create_train_test_validation_data()