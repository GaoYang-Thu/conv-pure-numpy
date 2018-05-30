# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:59:26 2018

@author: YangGao
"""
import numpy as np
from img_defs import create_train_data, process_test_data
from cnn_defs import conv, pooling, relu
import config_cov as cf
import cv2

if __name__ == '__main__':
    
    '''Data preparation'''
    training_data = create_train_data()
    test_data = process_test_data()
    
    # use single test image first
    img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
    
    '''Convolution output'''
    
    l1_filter = cf.L1_FILTER
    l1_output = relu(pooling(conv(img, l1_filter)))
    print('layer1 convolution is done \n')
    
    l2_shape = cf.L2_FILTER_SHAPE
    l2_filter = np.random.rand(l2_shape[0],l2_shape[1],l2_shape[2],l1_output.shape[-1])
    l2_output = relu(pooling(conv(l1_output, l2_filter)))
    print('layer2 convolution is done\n')
    
    l3_shape = cf.L3_FILTER_SHAPE
    l3_filter = np.random.rand(l3_shape[0],l3_shape[1],l3_shape[2],l2_output.shape[-1])
    l3_output = relu(pooling(conv(l2_output, l3_filter)))
    print('layer3 convolution is done\n')
    
