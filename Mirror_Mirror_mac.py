from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
import random
import io
import picamera
import tensorflow as tf
from scipy import ndimage
from six.moves import cPickle as pickle

# Data Path define
TRAIN_DATA_PATH = '/Users/JunChangWook/Tensorflow/Data/Mirror_Mirror_Data/Data'
REAL_DATA_PATH = '/Users/JunChangWook/Tensorflow/Data/Mirror_Mirror_Data/Predict_Data/real.jpg'
PREDICT_DATA_PATH = '/Users/JunChangWook/Tensorflow/Data/Mirror_Mirror_Data/Predict_Data/predict.jpg'
CASCADE_DATA_PATH = '/Users/JunChangWook/Tensorflow/Data/Process_Image/'

# Classification Number define
NUM_CLASSES = 2
# Image define
IMAGE_SIZE = 48
IMAGE_CHANNELS = 1
PIXEL_DEPTH = 255.0
# Train define
BATCH_SIZE = 160
PATCH_SIZE = 3
DEPTH = 16
NUM_HIDDEN = 64
NUM_STEPS = 30000

# Variable define

np.random.seed(133)

CASC_PATH = CASCADE_DATA_PATH + 'haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

# Create a memory stream so photos doesn't need to be saved in a file
stream = io.BytesIO()

# make pickle
def load_letter(folder, min_num_images):
    print('load_letter folder : %s min_num_images : %s' % (folder, min_num_images))
    image_files = os.listdir(folder)
    #print('image_files : %s' % image_files)
    dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        #print('image_file : %s' % (image_file))

        try:
            # Normalize
            image_data = (ndimage.imread(image_file).astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
            if image_data.shape != (IMAGE_SIZE , IMAGE_SIZE):
                print('Unexpected image shape: %s' % str(image_data.shape))
                continue
                #raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '-it\'s ok , skipping.')

    dataset = dataset[0:num_images, :, :]
    #if num_images < min_num_images:
    #    raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset

# folder => pickle
def maybe_pickle(data_folders, min_num_images_per_class, force=True):
    print('maybe_pickle data_folders : %s min_num_images_per_class : %s' % (data_folders, min_num_images_per_class))
    folder_list = os.listdir(data_folders)
    print('dir_list : %s' % folder_list)
    dataset_names = []
    for folder in folder_list:
        #print('folder %s' % folder)
        set_filename = folder + '.pickle'
        #print('set_filename %s' % set_filename)
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            #print('Pickling %s.' % set_filename)
            dataset = load_letter(TRAIN_DATA_PATH + '/' + folder, min_num_images_per_class)
            try:
                with open (set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names 

def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(pickle_files):
    num_classes = len(pickle_files)
    print('merge datasets num_classes : %s' % str(num_classes))
    train_size = 0
    start_t = 0
    end_t = 0

    for _, pickle_file in enumerate(pickle_files):
        try:
            #print('label : %s pickle_file : %s' % (label, pickle_file))
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                data_size = letter_set.shape[0]
                train_size += data_size
        except Exception as e:
            print('Unable to process data form', pickle_file, ':', e)
            raise

    print('make arrays train_size : %s' % str(train_size))
    train_datasets, train_labels = make_arrays(train_size, IMAGE_SIZE)
          
    for label, pickle_file in enumerate(pickle_files):
        try:
            print('label : %s pickle_file : %s' % (label, pickle_file))
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                data_size = letter_set.shape[0]
                end_t += data_size

                np.random.shuffle(letter_set)
                #Set Valid Data
                train_letter = letter_set[:, :, :]
                print('train_letter shape : %s' % str(train_letter.shape))
                train_datasets[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                print('train_dataset.shape : %s' % str(train_datasets.shape))
                print('train_labels.shape : %s' % str(train_labels.shape))

                start_t += end_t
        except Exception as e:
            print('Unable to process data form', pickle_file, ':', e)
            raise
    return train_datasets, train_labels

def reformat(dataset, labels):
    dataset = dataset.reshape(-1, IMAGE_SIZE * IMAGE_SIZE).astype(np.float32)
    labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
    return dataset, labels

# Data shuffle
'''def dataShuffle(dataset, labels):
    zip_data = list(zip(dataset, labels))
    random.shuffle(zip_data)
    dataset, labels = zip(zip_data)
    ran = random.random()
    random.shuffle(dataset, lambda : ran)
    random.shuffle(labels, lambda : ran)
    return dataset, labels

train_dataset, train_labels = dataShuffle(train_dataset, train_labels)
print('dataShuffle', train_dataset, train_labels)
print('dataShuffle.shape', train_dataset.shape, train_labels.shape)'''

# Data Channel Modify
def addchannelreformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)).astype(np.float32)
    return dataset, labels

# Model
def model(data, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
layer3_weights, layer3_biases, layer4_weights, layer4_biases, p_keep_input, p_keep_hidden):
    data = tf.nn.dropout(data, p_keep_input)
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer1_biases)
    hidden = tf.nn.dropout(hidden, p_keep_hidden)
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    hidden = tf.nn.relu(conv + layer2_biases)
    hidden = tf.nn.dropout(hidden, p_keep_hidden)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

# Error Display
def show_usage():
    print('Usage : python Mirror_Mirror.py')
    print('\t Mirror_Mirror.py train \t Trains and saves model with saved dataset')
    print('\t Mirror_Mirror.py poc \t\t Trains and  Launch the proof of concept')

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor = 1.3,
        minNeighbors = 5
    )        

    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC)
    except Exception:
        print("[+] Problem during resize")
        return None
    return image   

# make file list before classicfy 
def facecrop(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        img,
        scaleFactor = 1.3,
        minNeighbors = 5
    )    

    if not len(faces) > 0:
        return None
    print(faces)    
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    face = max_area_face
    img = img[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_CUBIC)
    return img    

# Get the picture(low resolution, so it should be quite fast)
def get_picture():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.capture(stream, format='jpeg')

        # Convert the picture into a numpy array
        buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

        # Now creates an OpenCV image
        image = cv2.imdecode(buff, 1)

        # Save the result image
        cv2.imwrite(REAL_DATA_PATH, image)

#  Main Funtions
if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_usage()
        exit()
    
    if sys.argv[1] == 'train':
        print('argv[1] : train')
        mTrain = True
    elif sys.argv[1] == 'poc':
        print('argv[1] : poc')
        mTrain = False
    else :
        show_usage()
        exit()        


    if mTrain:
        train_datasets = maybe_pickle(TRAIN_DATA_PATH, 2000)
        train_dataset, train_labels = merge_datasets(train_datasets)
        train_dataset, train_labels = reformat(train_dataset, train_labels)
        train_dataset, train_labels = addchannelreformat(train_dataset, train_labels)

    graph = tf.Graph()

    with graph.as_default():
        # Saver Init
        ckpt_dir = './ckpt_dir'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        global_step = tf.Variable(0, name='global_step', trainable=False)   

        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))

        # Variables.
        layer1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, IMAGE_CHANNELS, DEPTH], stddev=0.1))
        layer1_biases = tf.Variable(tf.zeros([DEPTH]))
        layer2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[DEPTH]))
        layer3_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * DEPTH, NUM_HIDDEN], stddev=0.1))
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[NUM_HIDDEN]))
        layer4_weights = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1))
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[NUM_CLASSES]))

        # Training computation.
        logits = model(tf_train_dataset, layer1_weights, layer1_biases, layer2_weights, layer2_biases, 
        layer3_weights, layer3_biases, layer4_weights, layer4_biases, 0.8, 0.5)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # Optimizer.
        #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
        #optimizer = tf.train.AdamOptimizer().minimize(loss)
        # Predictions for the training
        #train_prediction = tf.nn.softmax(logits)
        train_prediction = tf.argmax(logits, 1)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        if mTrain:    
            print('train')
            tf.initialize_all_variables().run()

            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path) # restore all variables
            start = global_step.eval() # get last global_step
            
            print("global_step :", start)
            for step in xrange(start, NUM_STEPS):
                offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
                batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
                feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
                _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                if (step % 100 == 0):
                    print('Minibatch loss at step ', step, ':', l)
                    global_step.assign(step).eval()
                    saver.save(session, ckpt_dir + '/model.ckpt', global_step=global_step)
        else:
            # Save picture
            get_picture()
            if(os.path.exists(REAL_DATA_PATH)):
                print('Full Filename : ' + REAL_DATA_PATH)
                img = cv2.imread(REAL_DATA_PATH)
                img = facecrop(img)
                print(img.shape)
                if len(img.shape) == 2:
                    print(img.shape)
                    image = img.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1])
                    print(image.shape)
                    print(image)
                    image = (image.astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
                    print(image)
                    image = np.float32(image)
                    print(type(image))
                    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(session, ckpt.model_checkpoint_path) # restore all variables
                        start = global_step.eval() # get last global_step
                        global_step.assign(start).eval()
                        logits = model(image, layer1_weights, layer1_biases, layer2_weights, 
                        layer2_biases, layer3_weights, layer3_biases, layer4_weights, layer4_biases, 1.0, 1.0)
                        #train_prediction = tf.argmax(logits, 1)
                        train_prediction = session.run(tf.argmax(logits, 1))
                        print('=====================================================')
                        print(train_prediction)
                        print('=====================================================')            
                    else:
                        print('saver data load Fail\t')
                else:
                    print('Prediction \t Fail')               