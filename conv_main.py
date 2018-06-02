# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:59:26 2018

@author: YangGao
"""
import numpy as np
from img_defs import create_train_data, process_test_data
from cnn_defs import conv2d, conv_single_pile, conv_pile_group, avg_pooling, relu, vector_and_concat, sigmoid, calculate_loss, generate_row_vector, generate_column_vector, reverse_vc, upsampling, paddle_zeros

import config_cov as cf
import cv2
import matplotlib.pyplot as plt



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
    filter_group_l2 = cf.L2_FILTER
    fc_weights = cf.FULLY_CONNECT_WEIGHTS
    l1_thresholds = cf.L1_THRESHOLDS
    l2_thresholds = cf.L2_THRESHOLDS
    fc_thresholds = cf.FULLY_CONNECT_THRESHOLDS
    
        # parameters for training and testing
    train_error = np.zeros(training_data.shape[0])
    epoch_num = cf.EPOCH_NUM
    learning_rate = cf.LEARNING_RATE

    for i in range(epoch_num):
        
        # train()?
        
        for img_index in range(training_data.shape[0]):
            
            '''forward throuth cnn'''
            img = training_data[img_index][0]
            img = img - np.mean(img)
            img = img / np.std(img)
            img_true_label = training_data[img_index][1]
            img_true_label = generate_column_vector(img_true_label) # convert it to a column vector
            
            # feed the image forward through the cnn
            l1_output_before_pooling = sigmoid(conv_single_pile(img,filter_pile_l1) + l1_thresholds)
            l1_output = avg_pooling(l1_output_before_pooling)
            
            l2_input = l1_output
            l2_output_before_activation = conv_pile_group(l2_input,filter_group_l2) + l2_thresholds
            l2_output_before_pooling = sigmoid(l2_output_before_activation)
            l2_output = avg_pooling(l2_output_before_pooling)
            fc_input_array = vector_and_concat(l2_output)
            fc_input_array = generate_column_vector(fc_input_array) # convert it to a column vector

            final_label_raw = np.dot(fc_weights,fc_input_array).reshape(cf.OUTPUT_LABLE_NUM,) + fc_thresholds
            final_label = sigmoid(final_label_raw)
            final_label = generate_column_vector(final_label) # convert it to a column vector
            
            train_error[img_index] = calculate_loss(img_true_label, final_label)
            
            if img_index %10 == 0:
                print('working on: image index: {}'.format(img_index))
            
            
            
            ''' backward probagation from loss function '''
            d_y = (img_true_label - final_label) * final_label * (1 - final_label)
            
            # d_fc_weights
            d_fc_weights = np.dot(d_y,fc_input_array.T)
            
            # d_fc_thresholds
            d_fc_thresholds = d_y
            
            d_fc_input_array = np.dot(fc_weights.T,d_y)
            d_pooling_layer2 = reverse_vc(d_fc_input_array)
            d_l2_output = upsampling(d_pooling_layer2)
            d_l2_output_before_activation = d_l2_output * l2_output_before_pooling * (1 - l2_output_before_pooling)
            
            # d_l2_filter_group
            d_l2_filter_group = cf.L2_FILTER # to ensure same shape
            (pile_num, filter_num) = d_l2_filter_group.shape[0],d_l2_filter_group.shape[1]
            for pile_index in range(pile_num):
                for filter_index in range(filter_num):
                    # flip Sp1 = l1_out_p, get d_l2_output_before_activation_q
                    l1_output_rot180_single = np.fliplr(np.flipud(l1_output[pile_index,:,:]))
                    d_l2_output_before_activation_single = d_l2_output_before_activation[filter_index,:,:]
                    # convolution
                    d_l2_filter_group[pile_index,filter_index,:,:] = conv2d(l1_output_rot180_single,d_l2_output_before_activation_single)


            # d_l2_thresholds
            d_l2_thresholds = cf.L2_THRESHOLDS
            (filter_num, filter_size) = d_l2_output_before_activation.shape[0], d_l2_output_before_activation.shape[1]
            for filter_index in range(filter_num):
                d_l2_thresholds[filter_index,:] = np.sum(d_l2_output_before_activation[filter_index,:,:]) # expand a 1x1 array to a 16x16 array
            
            d_l1_output = np.zeros((l1_output.shape))
            filter_num_l1 = cf.L1_FILTER_NUM
            filter_num_l2 = cf.L2_FILTER_NUM
            for pile_index in range(filter_num_l1):
                for filter_index in range(filter_num_l2):
                    # get single arrays for convolution
                    d_l2_output_before_activation_single = d_l2_output_before_activation[filter_index,:,:]
                    l2_filter_single = filter_group_l2[pile_index,filter_index,:,:]
                    # convolution
                    d_l1_output[pile_index,:,:] += conv2d(paddle_zeros(d_l2_output_before_activation_single), l2_filter_single)
            
            d_l1_output_before_pooling = upsampling(d_l1_output)
            
            d_l1_output_before_activation = d_l1_output_before_pooling * l1_output_before_pooling * (1 - l1_output_before_pooling)
           
            
            # d_l1_filter_pile
            d_l1_filter_pile = cf.L1_FILTER # same shape
            filter_num = d_l1_filter_pile.shape[0]
            for filter_index in range(filter_num):
                d_l1_output_before_activation_single = d_l1_output_before_activation[filter_index,:,:]
                img_rot180_single = np.fliplr(np.flipud(img))
                d_l1_filter_pile[filter_index,:,:] = conv2d(img_rot180_single, d_l1_output_before_activation_single)
                
            
        
            # d_l1_thresholds
            d_l1_thresholds = cf.L1_THRESHOLDS
            filter_num = d_l1_thresholds.shape[0]
            for filter_index in range(filter_num):
                d_l1_thresholds[filter_index,:,:] = np.sum(d_l1_output_before_activation[filter_index,:,:])
                
            
            ''' update all parameters!!!~ '''
            d_l1_filter_pile     -=   d_l1_filter_pile  * learning_rate
            d_l1_thresholds      -=   d_l1_thresholds   * learning_rate
            d_l2_filter_group    -=   d_l2_filter_group * learning_rate
            d_l2_thresholds      -=   d_l2_thresholds   * learning_rate
            d_fc_weights         -=   d_fc_weights      * learning_rate
            d_fc_thresholds      -=   d_fc_thresholds   * learning_rate
            
            
        # train() is completed without errors
            
            
            
            
    # validate
    for img_index in range(validating_data.shape[0]):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    