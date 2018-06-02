# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:59:26 2018

@author: YangGao
"""
import numpy as np
from img_defs import create_train_data, process_test_data
from cnn_defs import conv_single_pile, conv_pile_group, avg_pooling, relu, vector_and_concat, sigmoid, calculate_loss, generate_row_vector, generate_column_vector, reverse_vc, upsampling

import config_cov as cf
import cv2



if __name__ == '__main__':
    

    '''test '''
#    # use single test image first
#    img = cv2.imread('cat.jpg',cv2.IMREAD_GRAYSCALE)
#    img = cv2.resize(img,(50,50))
#    img = img - np.mean(img)
#    filter_pile_l1 = cf.L1_FILTER
#    filter_pile_l2 = cf.L2_FILTER
#    
#    l1 = conv_single_pile(img,filter_pile_l1)
#    l1_out = relu(avg_pooling(l1))
#    l2 = conv_pile_pile(l1_out,filter_pile_l2)
#    l2_out = relu(avg_pooling(l2))
#
#    
#    fc = vector_and_concat(l2_out)
#    
#    # fully connection
#    fc_weights = cf.FULLY_CONNECT_WEIGHTS
#    
#    final_y = np.dot(fc_weights, fc)
#    # no errors so far
    
    
    '''real convolution network'''
    
    ''' data '''
    training_data, validating_data = create_train_data()
    testing_data = process_test_data()
    
    ''' train '''
    
    # initialize parameters
        # paramters for cnn
    filter_pile_l1 = cf.L1_FILTER
    filter_pile_l2 = cf.L2_FILTER
    fc_weights = cf.FULLY_CONNECT_WEIGHTS
    l1_thresholds = cf.L1_THRESHOLDS
    l2_thresholds = cf.L2_THRESHOLDS
    fc_thresholds = cf.FULLY_CONNECT_THRESHOLDS
    
        # parameters for training and testing
    train_error = np.zeros(training_data.shape[0])
    epoch_num = cf.EPOCH_NUM
    learn_rate = cf.LEARNING_RATE

    for i in range(epoch_num):
        for img_index in range(training_data.shape[0]):
            
            '''forward throuth cnn'''
            img = training_data[img_index][0]
            img = img - np.mean(img)
            img = img / np.std(img)
            img_true_label = training_data[img_index][1]
            img_true_label = generate_column_vector(img_true_label) # convert it to a column vector
            
            # feed the image forward through the cnn
            l1_out = sigmoid(avg_pooling(conv_single_pile(img,filter_pile_l1) + l1_thresholds))
            l2_out = avg_pooling(sigmoid(conv_pile_group(l1_out,filter_pile_l2) + l2_thresholds))
            fc_input_array = vector_and_concat(l2_out)
            fc_input_array = generate_column_vector(fc_input_array) # convert it to a column vector

            final_label_raw = np.dot(fc_weights,fc_input_array) + fc_thresholds
            final_label = sigmoid(final_label_raw)
            final_label = generate_column_vector(final_label) # convert it to a column vector
            
            train_error[img_index] = calculate_loss(img_true_label, final_label)
            
            if img_index %10 == 0:
                print('working on: image index: {}'.format(img_index))
            
            
            
            ''' backward probagation from loss function '''
            d_y = (img_true_label - final_label) * final_label * (1 - final_label)
            
            d_fc_weights = np.dot(d_y,fc_input_array.T)
            d_fc_thresholds = d_y
            
            d_fc_input_array = np.dot(fc_weights.T,d_y)
            d_pooling_layer2 = reverse_vc(d_fc_input_array)
            d_l2_output = upsampling(d_pooling_layer2)
#            d_l2_filter_pile
#            d_l2_thresholds
#            d_l1_filter_pile
#            d_l1_thresholds
    
    # validate
    for img_index in range(validating_data.shape[0]):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    