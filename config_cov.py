# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:03:47 2018

@author: YangGao
"""
import numpy as np
from filter_generator_defs import filter_pile_generator, fully_connect_weights

'''
PATH
'''
TRAIN_PATH = 'K://Documents//Projects//machine-learning-basic//conv-pure-numpy//data//train'
TEST_PATH = 'K://Documents//Projects//machine-learning-basic//conv-pure-numpy//data//test'

'''
HYPERPARAMETERS
'''
LEARNING_RATE = 0.1

RELOAD_RAW_DATA = False # control whether to reload data from raw images
# True : Reload from raw images
# False: Don't reload. Only load from .npy files.

IMG_SIZE = 50
POOLING_SIZE = 2
POOLING_STRIDE = 2
VAL_FRACTION = 0.2
EPOCH_NUM = 1

# set filter_size of layer 1 and 2 according to input img size

L1_FILTER = filter_pile_generator(6,11)

L2_FILTER = filter_pile_generator(12,5)

FULLY_CONNECT_WEIGHTS = fully_connect_weights(2, 8*8*12)

