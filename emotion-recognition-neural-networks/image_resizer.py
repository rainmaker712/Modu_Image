#This file change the size of images

import os
import getopt
import sys
from PIL import Image

#Parsing the image
opts, args = getopt.getopt(sys.argv[1:], 'd:w:h')

# Set some default values to the needed variables.
directory = ''
width = -1
height = -1

#If an arguments was passed in, assign it to the correct variable.
for opt, arg in opts:
    if opt == '-d':
        directory = arg
    elif opt == '-w':
        width = int(arg)
    elif opt == '-h':
        height = int(arg)
#Make sure all arg passed

if width == -1 or height == -1 or directory == '';
    print('Invalid command line arguments. -d [directory] '
          '-w [width] -h [height] are required')

    #If an arg is missing exit the application
    exit()

#Iterate through every image given in the directory argument and resize it
for image in os.listdir(directory):
    print('Resizing image ' + image)

    # Open the image file
    img = Image.open(os.path.join(directory, image))
    # Resize it
    img = img.resize((width, height), Image.BILINEAR)
    # Save it back to disk.
    img.save(os.path.join(directory, 'resized-' + image))

print('Batch processing complete.')
