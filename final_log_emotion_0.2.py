#Ryan Shin: sungjin7127@gmail.com
#Date: 160927
#Obj: find the best model for emotion recogntion

from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import math
from six.moves import cPickle as pickle
from six.moves import range

import os
#%matplotlib inline
print ("Packages loaded")

#Load data
cwd = os.getcwd()
loadpath = cwd + "/dataset/data_gray.npz"
l = np.load(loadpath)

#Check what's included
print (l.files)

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
imgsize = l['imgsize']
#use_gray = l['use_gray']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("Image size is %s" % (imgsize))
print ("%d classes" % (nclass))

datasets = {
    "image_size": 64,
    "label_count": 8,
    "channel_count": 1
}
datasets["total_image_size"] = datasets["image_size"] * datasets["image_size"]

def reformat(dataset, labels, name):
    dataset = dataset.reshape((-1, datasets["image_size"], datasets["image_size"], datasets["channel_count"])).astype(np.float32)
    # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(datasets["label_count"]) == labels[:,None]).astype(np.float32)
    print(name + " set", dataset.shape, labels.shape)
    return dataset, labels
datasets["train"], datasets["train_labels"] = reformat(trainimg, trainlabel, "Training")
#datasets["valid"], datasets["valid_labels"] = reformat(valid_dataset, valid_labels, "Validation")
datasets["test"], datasets["test_labels"] = reformat(testimg, testlabel, "Test")

print(datasets.keys())

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def run_graph(graph_info, data, step_count, report_every=50):
    with tf.Session(graph=graph_info["graph"]) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        batch_size = graph_info["batch_size"]
        for step in xrange(step_count + 1):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (data["train_labels"].shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = data["train"][offset:(offset + batch_size), :, :, :]
            batch_labels = data["train_labels"][offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            targets = [graph_info["optimizer"], graph_info["loss"], graph_info["predictions"]]
            feed_dict = {graph_info["train"] : batch_data, graph_info["labels"] : batch_labels}
            _, l, predictions = session.run(targets, feed_dict=feed_dict)
            if (step % report_every == 0):
                print("Minibatch loss at step", step, ":", l)
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                #print("Validation accuracy: %.1f%%" % accuracy(graph_info["valid"].eval(), data["valid_labels"]))
        print("Test accuracy: %.1f%%" % accuracy(graph_info["test"].eval(), data["test_labels"]))


def convnet_two_layer(batch_size, patch_size, depth, hidden_size, data):
    image_size = data["image_size"]
    label_count = data["label_count"]
    channel_count = data["channel_count"]
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        train = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, channel_count))
        labels= tf.placeholder(tf.float32, shape=(batch_size, label_count))
        #valid = tf.constant(data["valid"])
        test  = tf.constant(data["test"])

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, channel_count, depth], stddev=0.1))
        layer1_biases  = tf.Variable(tf.zeros([depth]))
        layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
        layer2_biases  = tf.Variable(tf.constant(1.0, shape=[depth]))
        layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, hidden_size], stddev=0.1))
        layer3_biases  = tf.Variable(tf.constant(1.0, shape=[hidden_size]))
        layer4_weights = tf.Variable(tf.truncated_normal([hidden_size, label_count], stddev=0.1))
        layer4_biases  = tf.Variable(tf.constant(1.0, shape=[label_count]))

          # Model.
        def model(set):
            conv   = tf.nn.conv2d(set, layer1_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer1_biases)
            conv   = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
            hidden = tf.nn.relu(conv + layer2_biases)
            shape  = hidden.get_shape().as_list()
            reshape= tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
            hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
            return tf.matmul(hidden, layer4_weights) + layer4_biases

        # Training computation.
        logits = model(train)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        info = {
            "graph": graph,
            "batch_size": batch_size,
            "train": train,
            "labels": labels,
            "loss": loss,
            "optimizer": tf.train.GradientDescentOptimizer(0.05).minimize(loss),

            # Predictions for the training, validation, and test data.
            "predictions": tf.nn.softmax(logits),
            #"valid": tf.nn.softmax(model(valid)),
            "test":  tf.nn.softmax(model(test))
        }
    return info

graph_2conv = convnet_two_layer(batch_size=16, patch_size=5, depth=16, hidden_size=64, data=datasets)
run_graph(graph_2conv, datasets, 1000)    
