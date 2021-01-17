"""
    Experimentation with Inception V3 on a dataset of MRI covid/non-covid images.
"""

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from DataHandler import DataHandler

dh = DataHandler()
BATCH_SIZE = 512
dataset = dh.get_all_data_labeled(shuffle=True)
# for 80% train and 20% test split
split_point = int(len(dataset) * .8)
train_data = dataset[0:split_point]
test_data = dataset[split_point:]
train_data = train_data + train_data + train_data + train_data + train_data
random.shuffle(train_data)

train_X = np.asarray([dh.load_image(fp, resize=True, add_noise=True, randomly_rotate=True, grayscale=True)
                      for (fp, label) in train_data]).astype('uint8')
train_y = np.asarray([label for (fp, label) in train_data]).astype('float16')

test_X = np.asarray([dh.load_image(fp, resize=True, add_noise=True, randomly_rotate=True, grayscale=True)
                     for (fp, label) in test_data]).astype('uint8')
test_y = np.asarray([np.asarray(label) for (fp, label) in test_data]).astype('float16')

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
inception_model.trainable = False
x = inception_model.output
x = Dropout(0.2)(x)
x = Dense(32, activation='relu', dtype=tf.float32)(x)
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(16, activation='relu', dtype=tf.float32)(x)
x = Dropout(0.5)(x)
prediction = Dense(1, activation='sigmoid', dtype=tf.float32)(x)
complete_model = tf.keras.Model(inputs=inception_model.input, outputs=prediction)
complete_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=tf.keras.metrics.BinaryAccuracy())

# only the last layer should be trainable for now
complete_model.layers[len(complete_model.layers)-1].trainable = True
complete_model.fit(x=train_X, y=train_y, batch_size=BATCH_SIZE, validation_data=(test_X, test_y), epochs=320)
# inception model is now trained
inception_model.trainable = True
complete_model.fit(x=train_X, y=train_y, batch_size=BATCH_SIZE, validation_data=(test_X, test_y), epochs=50)
