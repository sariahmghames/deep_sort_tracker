import tensorflow as tf
from tensorflow.keras import layers, models

from config import Config

import datetime
import numpy as np
import os
import time




def pre_activation_residual_block(input_tensor, filters, kernel_size=(3, 3), strides=(1, 1)):
    # Applying BatchNormalization and Activation before the Convolutional layers
    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)

    # First Convolutional Layer
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    
    # Applying BatchNormalization and Activation before the second Convolutional layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second Convolutional Layer
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)

    # Adding the input to the output of the second convolutional layer (residual connection)
    x = Add()([x, input_tensor])
    return x



def residual_block(x, filters, kernel_size=(3, 3) ,strides=1, padding = 'same', use_projection=False):
    shortcut = x

    #print("shortcut shape = ", shortcut.shape)

    if use_projection: # based on paper on : wide residual networks
        # Apply 1x1 convolution to match the dimensions of the shortcut
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding=padding)(x)
        shortcut = layers.BatchNormalization()(shortcut)

    #print("shortcut shape = ", shortcut.shape)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    #print("x shape = ", x.shape)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    #print("x shape = ", x.shape)

    x = layers.add([x, shortcut])
    #x = layers.ReLU()(x)
    return x



def build_resnet(input_shape, num_classes, num_blocks, filters):
    inputs = layers.Input(shape=input_shape)

    #print("inputs = ", input_shape)

    x = layers.Conv2D(filters[0], (3, 3), strides = 1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    #print("layer 1 = ", x.shape)

    x = layers.Conv2D(filters[0], (3, 3), strides = 1, padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    #print("layer 2 = ", x.shape)

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2 , padding='same')(x)

    #print("layer 3 = ", x.shape)
    
    for i in range(2):
        x = residual_block(x, filters[1], kernel_size = (3, 3), strides= 1, padding ='same', use_projection=True)

    x = residual_block(x, filters[2], kernel_size = (3, 3), strides= 2, padding = 'same', use_projection=True)
    x = residual_block(x, filters[2], kernel_size = (3, 3), strides= 1, padding = 'same', use_projection=True)
    x = residual_block(x, filters[3], kernel_size = (3, 3), strides= 2, padding = 'same', use_projection=True)
    x = residual_block(x, filters[3], kernel_size = (3, 3), strides= 1, padding = 'same', use_projection=True)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    #x = layers.Dense(num_classes, activation='relu')(x)
    #x = layers.Dense(num_classes)(x)
    #x = layers.BatchNormalization()(x)
    #x = layers.LayerNormalization(axis=[1, 2, 3])


    model = models.Model(inputs, x) # forward propagation
    return model



if __name__ == '__main__':

    input_shape = (32, 32, 128, 3)  # Replace with the appropriate input shape for your dataset
    num_classes = 23          # Replace with the number of classes in your dataset
    num_blocks = 6           # Number of residual blocks in each stage
    filters = [64, 32, 16, 8]     # Initial number of filters, filters per stage, and number of stages

    cnn_model = build_resnet(input_shape, num_classes, num_blocks, filters)
    cnn_model.summary()




