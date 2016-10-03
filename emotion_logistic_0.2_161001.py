#Ryan Shin: sungjin7127@gmail.com
#Date: 160927

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print ("Packages loaded")

#Load data
cwd = os.getcwd()
loadpath = cwd + "/dataset/data_gray.npz"
l = np.load(loadpath)

#Check what's included
print (l.files)

#Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim = trainimg.shape[1]
ntest = testimg.shape[0]
print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimentional input" % (dim))
print ("%d classes" % (nclass))

#Define network
tf.set_random_seed(0)
#Param. of Log. Regression
learning_rate = 0.001
training_epochs = 1000
batch_size = 10
display_step = 100


with graph.as_default():
    graph = tf.Graph()
    #Saver init
    ckpt_dir = './ckpt_dir'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create Graph for Logistic Regression
    x = tf.placeholder("float", [None, dim])
    y = tf.placeholder("float", [None, nclass])
    W = tf.Variable(tf.zeros([dim, nclass]), name = 'weights')
    b = tf.Variable(tf.zeros([nclass]))

    #Define functions
    WEIGHT_DECAY_FACTOR = 0.000001
    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                       for v in tf.trainable_variables()])
    _pred = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(_pred),
                                        reduction_indices=1))
    cost = cost + WEIGHT_DECAY_FACTOR*l2_loss
    #Optimizer
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    _corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32))
    saver = tf.train.Saver()
    print ("Functions ready")

with tf.Session(graph=graph) as session:
    init = tf.initialize_all_variables().run()
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path) #restore all Variable
    start = global_step.eval() # get last globak

    print("global_step :", start)
    #Launch the graph
    #sess = tf.Session()
    #Optimize
    #sess.run(init)
    #Training Cycle
    for epoch in xrange(training_epochs):
        avg_cost = 0.
        num_batch = int(ntrain/batch_size)
        # Loop over all batches
        for i in xrange(num_batch):
            randidx = np.random.randint(ntrain, size=batch_size)
            batch_xs = trainimg[randidx, :]
            batch_ys = trainlabel[randidx, :]
            # Fit Training using batch data
            session.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            # Compute Average loss
            avg_cost += session.run(cost,
                                feed_dict = {x: batch_xs, y: batch_ys})/num_batch

            if (epoch % 10 == 0):
                print('Minbath loss at step ', epoch, ':', avg_cost)
                global_step.assign(epoch).eval()
                saver.save(session, ckpt_dir + '/model.ckpt', global_step=global_step)

            # Display logs per epoch step

            if epoch % display_step == 0:
                train_acc = session.run(accr, feed_dict={x: batch_xs, y: batch_ys})
                print ("Epoch: %03d/%03d cost: %.9f" %
                      (epoch, training_epochs, avg_cost))
                print (" Training accuracy: %.3f" % (train_acc))
                test_acc = session.run(accr, feed_dict={x: testimg, y: testlabel})
                print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished")
#sess.close()
print ("Session closed.")
