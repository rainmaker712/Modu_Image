#coding: utf-8

import time
import requests
import cv2
import operator
import numpy as np
import os

URL = 'https://api.projectoxford.ai/face/v1.0/detect'   #Detect Face Url
KEY = '2d3bf64a34da4f0e8af2b0ad5e4d632d'    #Login Key

#Save Path Define
DATA_PAHT = '/Users/JunChangWook/Tensorflow/Data/Mirror/data/'
OTHER_PATH = '/Users/JunChangWook/Tensorflow/Data/Mirror/other/'
EUNSOOK_PATH = '/Users/JunChangWook/Tensorflow/Data/Mirror/eunsook/' 

#Label Define
EUNSOOK_LABEL = 'eunsook'
OTHER_LABEL = 'other'


#Json Parser       
def processRequest(json, data, headers, params):
    result = None

    #while True:
    response = requests.request('post', URL, json = json, data = data, 
    headers = headers, params = params)

    print ('response.status_code : %d' % (response.status_code))
    print ('response.headers : %s'  % (response.headers))
    print ('response.content : %s' % (response.content)) 

    if response.status_code == 429:
        print('Message: %s' %( response.json()['error']['message']))
    elif response.status_code == 200 or response.status_code == 201:
        print('200 Message')
        if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
            result = None
        elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
            if 'application/json' in response.headers['content-type'].lower():
                result = response.json() if response.content else None
            elif 'image' in response.headers['content-type'].lower():
                result = response.content
    else:
        print('Error code : %d' % (response.status_code))
        print('Message : %s' % (response.json()['error']['message']))

    time.sleep(7)    
    return result            

# Save Image
def saveImage(img, label, filename, count):
    if label is not None:
        if label == EUNSOOK_LABEL:
            cv2.imwrite(EUNSOOK_PATH + str(count) + '_' + filename, img)
        elif label == OTHER_LABEL:
            cv2.imwrite(OTHER_PATH + str(count) + '_' + filename, img)
        
        print ('save complage : ') , (filename)            

# infomation face from cloud
def resultImage(result, img, filename):
    count = 0
    for infoFace in result:
        faceRectangle = infoFace['faceRectangle']
        cv2.rectangle(img, (faceRectangle['left'], faceRectangle['top']), 
        (faceRectangle['left'] + faceRectangle['width'], 
        faceRectangle['top'] + faceRectangle['height']), 
        color = (255, 255, 255), thickness =1)

        x = faceRectangle['left']
        y = faceRectangle['top']
        w = faceRectangle['width']
        h = faceRectangle['height']

        print ('x : %s y : %s w : %s h : %s' % (x, y, w, h))
        pointFace = img[y : y + h, x : x + w ]
        gender = infoFace['faceAttributes']['gender']
        print ('gender : %s'  % (gender))
        saveImage(pointFace, EUNSOOK_LABEL, filename, count)

        count = count + 1

def genderLabel(full_path, filename):
    pathToFileInDisk = full_path
    with open(pathToFileInDisk, 'rb') as f:
        data = f.read()

    # Face detection parameters
    params = { 'returnFaceAttributes': 'age,gender', 
               'returnFaceLandmarks': 'true'} 

    headers = dict()
    headers['Ocp-Apim-Subscription-Key'] = KEY
    headers['Content-Type'] = 'application/octet-stream'   

    json = None
    result = processRequest(json, data, headers, params)

    print ('result : %s' % (result))

    if result is not None:
        # Load the original image from disk
        datatouint8 = np.fromstring(data, np.uint8)
        img = cv2.imdecode(datatouint8, cv2.IMREAD_GRAYSCALE)
        resultImage(result, img, filename)

# make file list before classicfy 
def search(dirname):
    if os.path.isdir(dirname):
        filenames = os.listdir(dirname)
        ret=[]        

        for filename in filenames:
            fullfilename = os.path.join(dirname, filename)
            if os.path.isdir(fullfilename):
                ret = search(fullfilename)
            else:
                ext = os.path.splitext(fullfilename)[-1]
                if ext == '.jpg' or ext == '.JPG' :
                    t=[]
                    t.append(fullfilename)
                    t.append(filename)
                    ret.append(t)
        return ret                 



if __name__ == "__main__":
    pwd = DATA_PAHT
    ret = search(pwd)
    print (ret)

    for i in range(len(ret)):
        full_path = ret[i][0]
        filename = ret[i][1]
        print(full_path)
        print(filename)

        genderLabel(full_path, filename)
