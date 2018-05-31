# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:34:43 2018

Todo: this file contains data processing defs

@author: YangGao
"""

from tqdm import tqdm
import os
import numpy as np
import cv2
from random import shuffle
import config_cov as cf

train_dir = cf.TRAIN_PATH
test_dir = cf.TEST_PATH
img_size = cf.IMG_SIZE
reload_raw = cf.RELOAD_RAW_DATA # reload or not??
val_fraction = cf.VAL_FRACTION

def label_img(img_name):
    '''convert file names (cat.xx.jpg and dog.xx.jpg) to [1,0] and [0,1], respectively'''
    word_label = img_name.split('.')[0]
    if word_label == 'cat':
        return [1,0]
    elif word_label == 'dog':
        return [0,1]
    
def create_train_data():
    '''
    read images in training dataset directory,
    resize all images to the same size,
    extract label of each image ([1,0] = cat, [0,1] = dog),
    then formulate images and labels into a single array, trainging_data
    '''
    assert os.path.isdir(train_dir)
    
    training_data = []
    if not reload_raw:
        if os.path.exists('training_data.npy'):
            training_data = np.load('training_data.npy')
            print('training data loaded from .npy file')
    
    else:
        print('creating training data...')
        for img_name in tqdm(os.listdir(train_dir)):
            label = label_img(img_name)
            img_path = os.path.join(train_dir,img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_size,img_size))
            training_data.append([np.array(image),np.array(label)])
            shuffle(training_data)
            np.save('training_data.npy',training_data)
            
    training_data_size = training_data.shape[0]
    validation_data_size = int(val_fraction * training_data_size)
    validating_data = training_data[-validation_data_size:]
    training_data = training_data[:training_data_size-validation_data_size]
    print('validating data is created.')
    
    return training_data, validating_data




def process_test_data():
    '''
    read images in testing dataset directory,
    resize all images to the same size,
    extract number of each testing image,
    then formulate them into a single array, testing_data
    '''
    
    assert os.path.isdir(test_dir)
    testing_data = []
    
    if not reload_raw:
        if os.path.exists('testing_data.npy'):
            testing_data = np.load('testing_data.npy')
            print('testing data loaded from .npy file\n')
        
    else:

        print('creating test data...')
        for img_name in tqdm(os.listdir(test_dir)):
            img_path = os.path.join(test_dir, img_name)
            img_num = img_name.split('.')[0]
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_size,img_size))
            testing_data.append([np.array(image),img_num])
            shuffle(testing_data)
            np.save('testing_data.npy',testing_data)
    return testing_data