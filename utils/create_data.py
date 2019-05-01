import os
import random
from random import randint
import shutil
class CreateData():

	def __init__(self,data_folder):
		self.data_folder = data_folder
		self.train_folder = '../data/train/'
		self.test_folder = '../data/test/'
		self.val_folder = '../data/val/'


	def create_train_test_validation_data(self):
		self.categories = []
		all_files = []
		num_of_files=0

		for dirname, dirnames, filenames in os.walk(self.data_folder):
			# print path to all subdirectories first.
			for subdirname in dirnames:
				#print(os.path.join(dirname, subdirname))
				self.categories.append(subdirname)

			for filename in filenames:
				#print(filename)
				num_of_files=num_of_files+1
				all_files.append((dirname,filename))



		#print(all_files)
		test_val_sample_number = int(num_of_files * 0.2)
		print(num_of_files,test_val_sample_number)
		#test_val_sample = random.sample(all_files,test_val_sample_number)
		random.shuffle(all_files)


		for filename in all_files[:test_val_sample_number]:
			subdir=filename[0].split('/')[-1]

			if randint(0, 1):
				fn = self.test_folder+subdir+'/'+filename[1]
			else:
				fn = self.val_folder + subdir+'/' + filename[1]

			if not os.path.exists(os.path.dirname(fn)):
				try:
					os.makedirs(os.path.dirname(fn))
				except Exception as ex:
					print(ex)
					pass

			#os.rename(filename[0]+'/'+filename[1],fn)
			shutil.copy(filename[0]+'/'+filename[1],fn)

		for filename in all_files[test_val_sample_number:]:
			subdir = filename[0].split('/')[-1]
			fn = self.train_folder + subdir + '/' + filename[1]
			if not os.path.exists(os.path.dirname(fn)):
				try:
					os.makedirs(os.path.dirname(fn))
				except Exception as ex:
					print(ex)
					pass

			shutil.copy(filename[0] + '/' + filename[1], fn)


		print(self.categories)






