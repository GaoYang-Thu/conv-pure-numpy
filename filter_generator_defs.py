# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:16:37 2018

@author: YangGao
"""
import numpy as np


def filter_pile_generator(filter_num, filter_size):
    '''
    generate piles of filters for each convolution layer
    
    
    Arguments:
    filter_num --- number of filters in the pile
    filter_size -- size of each filter in the pile
    
    Returns:
    filter_pile -- a pile of filters
    '''
    
    filter_pile_shape = (filter_num, filter_size, filter_size)
    
    filter_pile = np.zeros(filter_pile_shape)
    
    for filter_index in range(filter_num):
        filter_pile[filter_index,:,:] = np.random.rand(filter_size,filter_size)
    
    print('the filter pile is generated.')
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
    
    fc_weights = np.random.randn(final_dim, array_length)
    print('the fc weights are generated.')
    
    return fc_weights
    
    
    
    
    
    
    