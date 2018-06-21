# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:59:26 2018

@author: YangGao
"""
import numpy as np
from img_defs import create_train_data_all, shuffle_and_create_validata_data, create_test_data_all
from cnn_defs import conv2d, conv_single_pile, conv_pile_group, avg_pooling, relu, vector_and_concat, sigmoid, calculate_loss, generate_row_vector, generate_column_vector, reverse_vc, upsampling, paddle_zeros, cnn_forward

import config_cov as cf
import cv2
import matplotlib.pyplot as plt


    

''' data '''
training_data_all = create_train_data_all()
testing_data = create_test_data_all()
epoch_num = cf.EPOCH_NUM
learning_rate = cf.LEARNING_RATE

validating_size = int(cf.VALIDATING_FRACTION * training_data_all.shape[0])
training_size = training_data_all.shape[0] - validating_size

training_error = np.zeros((training_size,epoch_num))
validating_error = np.zeros((validating_size,epoch_num))


''' initial parameters '''
l1_filter_pile = cf.L1_FILTER
l2_filter_group = cf.L2_FILTER
fc_weights = cf.FULLY_CONNECT_WEIGHTS
l1_thresholds = cf.L1_THRESHOLDS
l2_thresholds = cf.L2_THRESHOLDS
fc_thresholds = cf.FULLY_CONNECT_THRESHOLDS

print('Total epoch num = {}'.format(epoch_num))

''' train '''
for epoch_index in range(epoch_num):
        
    # split tratining_data_all into 2 subsets: training data and validating data
        
    print('Epoch index = {}'.format(epoch_index+1))
    print('Shuffling and spliting trainging data...\n')
    
    training_data, validating_data = shuffle_and_create_validata_data(training_data_all)
            
    # train()?
    print('\n')
    print('Training CNN......')
    
        
    for img_index in range(training_data.shape[0]):
            
        '''forward throuth cnn'''
        # summerize using a cnn_forward() funcion
        
        img = training_data[img_index][0]
        img = img - np.mean(img)
        img = img / np.std(img)
        img_true_label = training_data[img_index][1]
        img_true_label = generate_column_vector(img_true_label) # convert it to a column vector
            
        # feed the image forward through the cnn
        l1_output_before_pooling = sigmoid(conv_single_pile(img,l1_filter_pile) + l1_thresholds)
        l1_output = avg_pooling(l1_output_before_pooling)
            
        l2_input = l1_output
        l2_output_before_activation = conv_pile_group(l2_input,l2_filter_group) + l2_thresholds
        l2_output_before_pooling = sigmoid(l2_output_before_activation)
        l2_output = avg_pooling(l2_output_before_pooling)
        fc_input_array = vector_and_concat(l2_output)
        fc_input_array = generate_column_vector(fc_input_array) # convert it to a column vector
            
        fc_thresholds = generate_column_vector(fc_thresholds)
        final_label_raw = np.dot(fc_weights,fc_input_array) + fc_thresholds
        final_label = sigmoid(final_label_raw)
        final_label = generate_column_vector(final_label) # convert it to a column vector
            
        training_error[img_index,epoch_index] = calculate_loss(img_true_label, final_label)
        
        if (img_index+1) %10 == 0:
            print('Current index = {} / Total training_num = {} / Epoch index = {}'.format(img_index+1, training_size, epoch_index+1))
            
            
            
        ''' backward probagation from loss function '''
        # summerize using a cnn_backward() function
            
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
                l2_filter_single = l2_filter_group[pile_index,filter_index,:,:]
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
        l1_filter_pile     -=   d_l1_filter_pile  * learning_rate
        l1_thresholds      -=   d_l1_thresholds   * learning_rate
        l2_filter_group    -=   d_l2_filter_group * learning_rate
        l2_thresholds      -=   d_l2_thresholds   * learning_rate
        fc_weights         -=   d_fc_weights      * learning_rate
        fc_thresholds      -=   d_fc_thresholds   * learning_rate
            
        # save parameters???
            
    print('Training CNN is completed. \n')
    # train() is completed without errors
            
        
    # validate     
    print('Validating CNN......')
        
    for img_index in range(validating_size):
            
        # feed each img in validating data through cnn
        img = validating_data[img_index][0]
        img = img - np.mean(img)
        img = img / np.std(img)
        img_true_label = generate_column_vector(validating_data[img_index][1])
            
        l1_output_validating = avg_pooling(sigmoid(conv_single_pile(img,l1_filter_pile) + l1_thresholds))
        l2_output_validating = avg_pooling(sigmoid(conv_pile_group(l1_output_validating, l2_filter_group) + l2_thresholds))
            
        fc_input_validating = generate_column_vector(vector_and_concat(l2_output_validating))
        final_label_validating = sigmoid(np.dot(fc_weights,fc_input_validating) + fc_thresholds)                     
        final_label_validating = generate_column_vector(final_label_validating)
            
        # calculate loss and store it in validate_error
        validating_error[img_index,epoch_index] = calculate_loss(img_true_label, final_label_validating)
            
        if (img_index+1) %10 == 0:
            print('Current validating index = {} / Total validating_num = {} / Epoch index = {}'.format(img_index+1, validating_size, epoch_index+1))
    
    print('Validating is completed. \n')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    