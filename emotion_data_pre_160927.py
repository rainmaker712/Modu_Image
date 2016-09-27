#Ryan Shin: sungjin7127@gmail.com
#Date: 160927
#Obj: load all the data and do image preprocessing + convert into npz dataset

"""1. convert paths variable into your own image folder
   2. convert imgsize into your desired size
 """

#Basic Generating Dataset
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
#%matplotlib inline
print("Package loaded")
cwd = os.getcwd()
print ("Current folder is %s" % (cwd) )

################################################################################
#Specify folder paths + reshape size + grayscale
#Training Set folder: change this
paths = {"dataset/anger/", "dataset/contempt/", "dataset/disgust/", "dataset/fear",
        "dataset/happy/", "dataset/neutral/", "dataset/sadness/", "dataset/surprise/"}
#The reshape size
imgsize = [64, 64]
#Grayscale
use_gray = 1
#Save name
data_name = "data_gray"
######################################################################################3

print ("Your image should be at")
for i, path in enumerate(paths):
    print (" [%d/%d] %s/%s" % (i, len(paths), cwd, path))

print ("Data will be save to %s"
      % (cwd + '/data/' + data_name + '.npz'))


#RGB2Gray Fucn.
def rgb2gray(rgb):
    if len(rgb.shape) is 3:
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    else:
        print("Current Image if Gray")
        return rgb

#Load Image
nclass = len(paths)
valid_exts = [".jpg",".gif",".png",".tga",".jpeg"]
imgcnt = 0
for i, relpath in zip(range(nclass), paths):
    path = cwd + "/" + relpath
    flist = os.listdir(path)
    for f in flist:
        if os.path.splitext(f)[1].lower() not in valid_exts:
            continue
        fullpath = os.path.join(path, f)
        currimg = imread(fullpath)
        #Convert to grayscale
        if use_gray:
            grayimg = rgb2gray(currimg)
        else:
            grayimg = currimg
        # Reshape
        graysmall = imresize(grayimg, [imgsize[0], imgsize[1]])/255.
        grayvec = np.reshape(graysmall, (1, -1))
        # Save
        curr_label = np.eye(nclass, nclass)[i:i+1, :]
        if imgcnt is 0:
            totalimg = grayvec
            totallabel = curr_label
        else:
            totalimg = np.concatenate((totalimg, grayvec), axis = 0)
            totallabel = np.concatenate((totallabel, curr_label), axis = 0)
        imgcnt = imgcnt + 1
print ("Total %d images loaded." % (imgcnt))

#Divide total Data into Training and Test Set
def print_shape(string, x):
    print ("Shape of '%s' is %s" % (string, x.shape,))

randidx = np.random.randint(imgcnt, size=imgcnt)
trainidx = randidx[0:int(3*imgcnt/5)]
testidx = randidx[int(3*imgcnt/5):imgcnt]
trainimg = totalimg[trainidx, :]
trainlabel = totallabel[trainidx, :]
testimg = totalimg[testidx, :]
testlabel = totallabel[testidx, :]
print_shape("trainimg", trainimg)
print_shape("trainlabel", trainlabel)
print_shape("testimg", testimg)
print_shape("testlabel", testlabel)

#Save to Npz
savepath = cwd + "/dataset/" + data_name + ".npz"
np.savez(savepath, trainimg=trainimg, trainlabel=trainlabel,
        testimg = testimg, testlabel = testlabel, imgsize=imgsize)
print ("Saved to %s" % (savepath))

#Load
cwd = os.getcwd()
loadpath = cwd + "/dataset/" + data_name + ".npz"
l = np.load(loadpath)

#See what's in here
l.files

#Parse data
trainimg_loaded = l['trainimg']
trainlabel_loaded = l['trainlabel']
testimg_loaded = l['testimg']
testlabel_loaded = l['testlabel']

print ("%d train image loaded" % (trainimg_loaded.shape[0]))
print ("%d test images loaded" % (testimg_loaded.shape[0]))
print ("Loaded from to %s" % (savepath))
