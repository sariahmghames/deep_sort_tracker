import os
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import cv2


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
from deepsort_resnet import *



cfg = Config()
input_shape = cfg.input_shape
num_classes = cfg.num_classes
num_filters = cfg.num_filters
num_blocks = cfg.num_blocks
batch_size = cfg.batch_size
epochs = cfg.epochs
learning_rate = cfg.learning_rate
training_dir = cfg.training_dir
testing_dir = cfg.testing_dir
dataset_dir = cfg.dataset_dir


print("GPU compute available: ", tf.test.is_gpu_available())




def create_data_generators(train_images, train_labels, test_images, test_labels, batch_size=32):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Normalize pixel values to [0, 1]
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
    test_generator = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

    return train_generator, test_generator


def create_train_test_sets(images, labels, test_size=0.2, random_state=42):
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)


def load_and_preprocess_data(directory, img_size=(150, 150)):
    images = []
    labels = []

    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = cv2.resize(img, img_size)  # Resize to the desired size, (width, height)
        images.append(img)
        labels.append(int(image_name.split('.')[0].split('_')[1]))

    return np.array(images), np.array(labels)



# Train your model 
def train_model(model, train_data, validation_data):
    cnn_model = model.fit(train_data, epochs=epochs, batch_size=batch_size,
                        validation_data=validation_data)
    return cnn_model



if __name__ == "__main__":

	# Step 1: Load and preprocess data
	images, labels = load_and_preprocess_data(dataset_dir, img_size =(150, 300)) 

	# Step 2: Create train and test sets
	train_data, test_data, train_labels, test_labels = create_train_test_sets(images, labels)

	print("train data shape=", train_data.shape)
	print("test data shape = ", test_data.shape)

	steps_per_epoch = len(train_data)//batch_size


	# Step 3: Create data generators that yields batches of data and labels
	train_generator, test_generator = create_data_generators(train_data, train_labels, test_data, test_labels)

	#  build your model and print it out
	cnn_model = build_resnet(input_shape, num_classes, num_blocks, num_filters)
	cnn_model.summary()

	# compile your model
	cnn_model.compile(loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' if your labels are integers
	              optimizer='Adam',  
	              metrics=['accuracy'])

	# Train your model
	cnn_model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
	#cnn_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)


	loss, accuracy = cnn_model.evaluate(test_data, test_labels)
	print("Test Loss:", loss)
	print("Test Accuracy:", accuracy)


# Loss curve plot to be dumped after full model training. 
#show_plot(counter,loss_history,path='ckpts/loss.png')
