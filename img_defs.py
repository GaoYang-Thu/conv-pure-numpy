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
validating_fraction = cf.VALIDATING_FRACTION




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
    print('\n')
    print('Loading training data......')
    training_data_all = []
    
    # load all training data
    if not reload_raw:
        if os.path.exists('training_data_all.npy'):
            training_data_all = np.load('training_data_all.npy')
            
            print('all training data loaded from .npy file')
    
    else:
        print('creating training data...')
        for img_name in tqdm(os.listdir(train_dir)):
            label = label_img(img_name)
            img_path = os.path.join(train_dir,img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_size,img_size))
            training_data_all.append([np.array(image),np.array(label)])
            shuffle(training_data_all)
            np.save('training_data_all.npy',training_data_all)
    
    training_validating_size = training_data_all.shape[0]
    
    validation_data_size = int(validating_fraction * training_validating_size)
    validating_data = training_data_all[-validation_data_size:]
    
    training_data_size = training_validating_size - validation_data_size
    training_data = training_data_all[:training_data_size]
    
    print('Training + Validating = {} data points'.format(training_validating_size))
    print('Training              = {} data points'.format(training_data_size))
    print('Validating            = {} data points'.format(validation_data_size))
    
    print('image size = {} * {} \n'.format(img_size, img_size))
    
    return training_data, validating_data, training_data_size




def process_test_data():
    '''
    read images in testing dataset directory,
    resize all images to the same size,
    extract number of each testing image,
    then formulate them into a single array, testing_data
    '''
    
    assert os.path.isdir(test_dir)
    
    print('Loading tesing data......')
    testing_data = []
    
    # load all tesing data
    if not reload_raw:
        if os.path.exists('testing_data_all.npy'):
            testing_data = np.load('testing_data_all.npy')
            print('all testing data loaded from .npy file')
        
    else:

        print('creating test data...')
        for img_name in tqdm(os.listdir(test_dir)):
            img_path = os.path.join(test_dir, img_name)
            img_num = img_name.split('.')[0]
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (img_size,img_size))
            testing_data.append([np.array(image),img_num])
            shuffle(testing_data)
            np.save('testing_data_all.npy',testing_data)
    
    testing_data_size = cf.TESTING_DATA_SIZE
    testing_data = testing_data[:testing_data_size]
    print('Testing = {} data points \n'.format(testing_data_size))
    
    return testing_data









