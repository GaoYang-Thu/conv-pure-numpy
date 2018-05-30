# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:03:47 2018

@author: YangGao
"""

import numpy as np

'''
PATH
'''
TRAIN_PATH = 'K://Documents//Projects//machine-learning-basic//conv-pure-numpy//data//train'
TEST_PATH = 'K://Documents//Projects//machine-learning-basic//conv-pure-numpy//data//test'

'''
HYPERPARAMETERS
'''
LEARNING_RATE = 1E-3

RELOAD_RAW_DATA = False # control whether to reload data from raw images
# True : Reload from raw images
# False: Don't reload. Only load from .npy files.

IMG_SIZE = 50
POOLING_SIZE = 2
POOLING_STRIDE = 2
EPOCH_NUM = 10

def l1_filter():
    L1_FILTER_SHAPE = (2,3,3)
    L1_FILTER = np.zeros(L1_FILTER_SHAPE)
    L1_FILTER[0,:,:] = np.array([[-1,0,1],
                                [-1,0,1],
                                [-1,0,1]])
    L1_FILTER[1,:,:] = np.array([[1,1,1],
                                [0,0,0],
                                [-1,-1,-1]])
    return L1_FILTER

L1_FILTER = l1_filter()

L2_FILTER_SHAPE = np.array([3,5,5])

L3_FILTER_SHAPE = np.array([1,7,7])