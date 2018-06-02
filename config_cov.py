# -*- coding: utf-8 -*-
"""
Created on Tue May 29 15:03:47 2018

@author: YangGao
"""
import numpy as np
from filter_generator_defs import filter_pile_generator_l1, fully_connect_weights, threshold_generator, filter_pile_generator_l2

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
OUTPUT_LABLE_NUM = 2

VAL_FRACTION = 0.2
EPOCH_NUM = 1

# layer 1 filter pile and thresholds

L1_FILTER_NUM = 6
L1_FILTER_SIZE = 11
L1_FILTER = filter_pile_generator_l1(L1_FILTER_NUM, L1_FILTER_SIZE)
L1_FILTER_OUTPUT_SIZE_BEFORE_POOLING = int(IMG_SIZE - L1_FILTER_SIZE + 1)
L1_THRESHOLDS = threshold_generator(L1_FILTER_NUM, L1_FILTER_OUTPUT_SIZE_BEFORE_POOLING)
L1_FILTER_OUTPUT_SIZE_AFTER_POOLING = int(L1_FILTER_OUTPUT_SIZE_BEFORE_POOLING / 2)  # AFTER POOLING


# layer 2 filter pile and thresholds

L2_FILTER_NUM = 12
L2_FILTER_SIZE = 5
L2_FILTER = filter_pile_generator_l2(L1_FILTER_NUM, L2_FILTER_NUM, L2_FILTER_SIZE)
L2_FILTER_OUTPUT_SIZE_BEFORE_POOLING = int(L1_FILTER_OUTPUT_SIZE_AFTER_POOLING - L2_FILTER_SIZE + 1)
L2_THRESHOLDS = threshold_generator(L2_FILTER_NUM, L2_FILTER_OUTPUT_SIZE_BEFORE_POOLING)
L2_FILTER_OUTPUT_SIZE_AFTER_POOLING = int(L2_FILTER_OUTPUT_SIZE_BEFORE_POOLING / 2) # AFTER POOLING


# fully connect layer weights and thresholds

FULLY_CONNECT_ARRAY_LEN = L2_FILTER_OUTPUT_SIZE_AFTER_POOLING * L2_FILTER_OUTPUT_SIZE_AFTER_POOLING * L2_FILTER_NUM
FULLY_CONNECT_WEIGHTS = fully_connect_weights(OUTPUT_LABLE_NUM, FULLY_CONNECT_ARRAY_LEN)

FULLY_CONNECT_THRESHOLDS = np.random.randn(OUTPUT_LABLE_NUM)
