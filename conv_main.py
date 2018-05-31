# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:59:26 2018

@author: YangGao
"""
import numpy as np
from img_defs import create_train_data, process_test_data
from cnn_defs import conv_pile, avg_pooling, relu, sigmoid
import config_cov as cf
import math
import cv2

epoch_num = cf.EPOCH_NUM

if __name__ == '__main__':
    

    '''test '''
#    # use single test image first
#    img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
#    
#    l1_filter = cf.L1_FILTER
#    l1_output = relu(pooling(conv(img, l1_filter)))
#    print('layer1 convolution is done \n')
#    
#    l2_shape = cf.L2_FILTER_SHAPE
#    l2_filter = np.random.rand(l2_shape[0],l2_shape[1],l2_shape[2],l1_output.shape[-1])
#    l2_output = relu(pooling(conv(l1_output, l2_filter)))
#    print('layer2 convolution is done\n')
#    
#    l3_shape = cf.L3_FILTER_SHAPE
#    l3_filter = np.random.rand(l3_shape[0],l3_shape[1],l3_shape[2],l2_output.shape[-1])
#    l3_output = relu(pooling(conv(l2_output, l3_filter)))
#    print('layer3 convolution is done\n')
    
    '''real convolution network'''
    
    # data
    training_data = create_train_data()
    training_data_size = training_data.shape[0]
    
    validation_data_size = int(cf.VAL_FRACTION * training_data_size)
    validating_data = training_data[-validation_data_size:]
    training_data = training_data[:training_data_size-validation_data_size]
    
    testing_data = process_test_data()
    
    # train
    for i in range(epoch_num):
        for img_index in range(training_data.shape[0]):
            img = training_data[img_index][0]
            img_true_label = training_data[img_index][1]
            
            # feed the image forward through the cnn
            
            # backward from loss function
            
    # validate
    for img_index in range(validating_data.shape[0]):
        pass
    
    
    