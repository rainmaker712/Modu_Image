from __future__ import division, print_function, absolute_import
import cv2
import random
import numpy as np
from shutil import copyfile
import glob

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotion order
#participants = glob.glob("source_emotion/*") #Returns a list of all folders with participant numbers

emotion = 'surprise'

ddata = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    #files = glob.glob("dataset_test/%s/*" %emotion)
    files = glob.glob("dataset_test/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            training_data.append(gray) #append image array to training data list
            training_labels.append(emotions.index(emotion))

        for item in prediction: #repeat above process for prediction set
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

training_data, training_labels, prediction_data, prediction_labels = make_sets()

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

#import tflearn
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

#Convert all data into numpy
X, Y, X_test, Y_test = np.array(training_data), np.array(training_labels), np.array(prediction_data), np.array(prediction_labels)

#Shuffling and one hot encoding

X, Y = shuffle(X,Y)

#Y = to_categorical(Y, 8)
#Y_test = to_categorical(Y_test, 8)

def dense_to_one_hot(labels_dense, num_classes=8):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

Y = dense_to_one_hot(Y)
Y_test = dense_to_one_hot(Y_test)

#Convert dataset into...
IMAGE_SIZE = 48

X = X.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])
X_test = X_test.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])

# Building convolutional network
network = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
network = conv_2d(network, 48, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 96, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 192, activation='tanh')
network = fully_connected(network, 384, activation='tanh')
network = dropout(network, 0.8)
network = dropout(network, 0.8)
network = fully_connected(network, 8, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=50,
           validation_set=({'input': X_test}, {'target': Y_test}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
