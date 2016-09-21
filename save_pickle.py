#Date: 0916
#Name: Seongjin Shin
#Email: sungjin7127@gmail.com
#Description
from __future__ import print_function
import numpy as np
import os
import sys
import tarfile
import sys

from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from scipy import ndimage

"""1. Save folder list in list """
#TODO: verfity change num_classes

num_classes = 8
np.random.seed(133)

def folder_list(folder_name):
    data_folders = [
        os.path.join(folder_name, d) for d in sorted(os.listdir(folder_name))
            if os.path.isdir(os.path.join(folder_name, d))]
    if len(data_folders) != num_classes:

        raise Exception(
          'Expected %d folders, one per class. Found %d instead.' % (
            num_classes, len(data_folders)))
    return data_folders
#TODO: Input folder name
train = 'dataset'
#test = 'RafD_test'
train_folders = folder_list(train)
#test_folders = folder_list(test)

#TODO: Change image_size
image_size = 350  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

"""2. Save data into pixels """
def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  image_index = 0
  print(folder)
  for image in os.listdir(folder):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[image_index, :, :] = image_data
      image_index += 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  num_images = image_index
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

#TODO: set variable name and check number of files
train_datasets = maybe_pickle(train_folders, 350)
#test_datasets = maybe_pickle(test_folders, 200)


"""3. Check the data to be balanced across classes """
def disp_number_images(data_folders):
    for folder in data_folders:
        pickle_filename = ''.join(folder) + '.pickle'
        try:
            with open(pickle_filename, 'r') as f:
                dataset = pickle.load(f)
        except Exception as e:
            print('Unable to read data from', pickle_filename, ':', e)
            return
        print('Number of images in ', folder, ' : ', len(dataset))

#TODO: check dataset is balanced
disp_number_images(train_folders)
#disp_number_images(test_folders)
