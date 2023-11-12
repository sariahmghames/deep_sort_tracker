import os


class Config():

	def __init__(self,):

		# Model Hyperparameters
		self.input_shape = (64, 128, 3)  # Example input shape for an image (32x32 with 3 channels), widthxheightxchannel
		self.num_classes = 23  # Number of classes in the classification problem

		# Architecture Hyperparameters
		self.num_filters = [128, 64, 32, 16]   # Number of filters in the convolutional layers
		self.num_blocks = 6  # Number of residual blocks in the model

		# Training Hyperparameters
		self.batch_size = 32  # Batch size for training
		self.epochs = 500  # Number of training epochs
		self.learning_rate = 0.001  # Learning rate for optimizer (e.g., Adam)

		self.training_dir = "../data/darko/bbox_train/"
		self.testing_dir = "../data/darko/bbox_test/"
		self.dataset_dir = "../data/darko/dataset/"
