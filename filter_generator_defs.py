# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:16:37 2018

@author: YangGao
"""
import numpy as np
import config_cov as cf





def filter_pile_generator_l1(filter_num, filter_size):
    '''
    generate piles of filters for convolution layer1
    
    
    Arguments:
        filter_num --- number of filters in the pile
        filter_size -- size of each filter in the pile
    
    Returns:
        filter_pile -- a pile of filters
    '''
    
    filter_length = filter_size ** 2
    filter_pile_shape = (filter_num, filter_size, filter_size)
    
    filter_pile = np.zeros(filter_pile_shape)
    
    low = - np.sqrt(filter_num/((1 + filter_num) * filter_size**2))
    
    for filter_index in range(filter_num):
        filter_pile[filter_index,:,:] = np.random.uniform(low=low, high=-low, size = filter_length).reshape(filter_size, filter_size)
    
    
    print('\n')
    print('Building CNN......')
    print('the filter pile for layer 1 is generated.')
    return filter_pile









def filter_pile_generator_l2(pile_num, filter_num, filter_size):
    '''
    generate piles of filters for convolution layer1
    
    
    Arguments:
        pile_num
        filter_num --- number of filters in the pile
        filter_size -- size of each filter in the pile
    
    Returns:
        filter_pile -- a pile of filters, shape = (pile_num, l2_input_num, filter_size, filter_size) 4d array
    '''
    filter_length = filter_size ** 2
    
    filter_pile_shape = (pile_num, filter_num, filter_size, filter_size)
    
    filter_pile = np.zeros(filter_pile_shape)
    
    low = - np.sqrt(pile_num/((pile_num + filter_num) * filter_size**2))
    
    for pile_index in range(pile_num):
        for filter_index in range(filter_num):
            filter_pile[pile_index, filter_index,:,:] = np.random.uniform(low=low, high=-low, size = filter_length).reshape(filter_size,filter_size)
            
    
    print('the filter piles for layer 2 is generated.')
    return filter_pile






def fully_connect_weights(final_dim, array_length):
    '''
    generate weights for fully connection layer
    
    Arguments:
        final_dim --- dimension of the final output array (eg. final_dim = 2 in cats vs dogs)
        array_length -- length of the input array of fully connection layer
    
    Returns:
        fc_weights -- a 2d array of weights for fully connection layer
    '''
    
    fc_weights_shape = (final_dim, array_length)
    total_length = final_dim * array_length
    filter_num = cf.L1_FILTER_NUM
    
    low = - np.sqrt(filter_num/(final_dim + array_length))
    
    fc_weights = np.random.uniform(low = low, high = -low, size = total_length).reshape(fc_weights_shape)
    
    print('the fc weights are generated.')
    
    return fc_weights
    










def threshold_generator(threshold_num, filter_size):
    '''
    generate thresholds for each convolution layer
    
    
    Arguments:
        threshold_num
        filter_size
    
    Returns:
        threshold array -- length = threshold_number
    '''
    
    threshold_array_shape = (threshold_num, filter_size, filter_size)
    
    threshold_array = np.zeros(threshold_array_shape)
    
#    for threshold_index in range(threshold_num):
#        threshold_array[threshold_index] += np.random.randint(0,10 + 1)
    
    print('the thresholds for current layer are generated.')
    
    return threshold_array
    
    
    
    
    