# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:23:01 2018

@author: YangGao
"""

import numpy as np
from scipy import signal
import sys
import config_cov as cf

filter_1 = cf.L1_FILTER


def conv2d(img_single, filter_single):
    
    '''

    2d convolution of a single image and a single filter
    
    Arguments:
    img_single -- single image to convolute, shape = (img_size_h, img_size_w), GRAY image
    filter_single ---- convolution kernel weights, square np array, shape = (filter_size, filter_size)
    
    Returns:
    result -- conv result, shape = (filter_number, img_size_h_cov, img_size_w_cov)
    cache --- cache of values for conv_backward()
    '''


    filter_size = filter_single.shape[0]
    (img_size_h, img_size_w) = img_single.shape
    img_size_h_cov = img_size_h - filter_size + 1
    img_size_w_cov = img_size_w - filter_size + 1
    result_single = np.zeros((img_size_h_cov, img_size_w_cov))
    
    for h in range(img_size_h_cov):
        for w in range(img_size_w_cov):
            img_slice = img_single[h:h + filter_size,
                                   w:w + filter_size]
            
            # remember to flip the kernel for convolution
            conv_filter_fliped = np.flipud(np.fliplr(filter_single))
            
            result_single[h,w] = np.sum(img_slice * conv_filter_fliped)
    
    cache_single = (img_single, filter_single)
    return result_single, cache_single
    '''
    scipy offers 2d method: signal.convolve2d
    I compared the results:
    r1 = conv2d(img, filter_1[0,:,:]) (my method)
    r3 = signal.convolve(img, filter_1[0,:,:],'valid') (scipy method)
    by using
    np.testing.assert_array_almost_equal(r1,r3)
    no errors were given, thus r1 and r3 are almost the same
    r1 - r3 is not 0, but the difference is fairly small to pass the 'almost same' array test
    '''
    
def conv2d_backward(d_result_single, cache_single):
    '''
    backward of 2d convolution of a single image and a single filter
    
    Arguments:
    d_result_single -- gradient of the loss with respect to the output of conv2d
                       shape = (img_size_h_cov, img_size_w_cov)
    cache ------------ output of conv2d()
    
    Returns:
    d_img_single ----- gradient of the loss with respect to the input of conv2d, 
                    shape = (img_size_h, img_size_w)
    d_filter_single -- gradient of the loss with respect to the filter of conv2d, 
                    shape = (filter_size, filter_size)
    '''
    
    (img_single, filter_single) = cache_single
    
    (img_size_h, img_size_w) = img_single.shape
    (filter_size, filter_size) = filter_single.shape
    (img_size_h_cov, img_size_w_cov) = d_result_single.shape
    
    d_img_single = np.zeros(img_single.shape)
    d_filter_single = np.zeros(filter_single.shape)
    
    for h in range(img_size_h_cov):
        for w in range(img_size_w_cov):
            
            d_img_single[h:h+filter_size, w:w+filter_size] += filter_single * d_result_single[h,w]
            
            d_filter_single_raw = img_single[h:h+filter_size, w:w+filter_size] * d_result_single[h,w]
            # flip back
            d_filter_single += np.fliplr(np.flipud(d_filter_single_raw))
    
    return d_img_single, d_filter_single
    
    

    
def conv_pile(img_single, filter_pile):
    '''
    2d convolution of a gray image and a pile of convolution filters
    
    Arguments:
    img_single ----- a image to conv, 
                   shape = (img_size_h, img_size_w)
    filter_pile -- a pile of filters, 
                   shape = (filter_num, filter_size, filter_size)
    
    Returns:
    feature_map_pile -- a plie of convolution results,
                        shape = (filter_num, img_size_h_cov, img_size_w_cov)
        
    '''
            
    filter_num = filter_pile.shape[0]
    filter_size = filter_pile.shape[1]
    img_size_h = img_single.shape[0]
    img_size_w = img_single.shape[1]
    img_size_h_cov = img_size_h - filter_size + 1
    img_size_w_cov = img_size_w - filter_size + 1
    
    feature_map_pile = np.zeros((filter_num,
                                 img_size_h_cov,
                                 img_size_w_cov))
    
    for filter_index in range(filter_num):
        
        print('Using filter number {}...'.format(filter_index + 1))
        
        curr_filter = filter_pile[filter_index,:,:]
        
        feature_map_single = conv2d(img_single, curr_filter)[0] # we want only the conv result
        
        feature_map_pile[filter_index,:,:] = feature_map_single
    
    return feature_map_pile
    
def avg_pooling(feature_map_pile, 
                pool_size = cf.POOLING_SIZE, 
                stride = cf.POOLING_STRIDE):
    '''
    average pooling (= 75% downsampling when pool_size = stride = 2)
    
    Arguments:
    feature_map_pile -- image pile after convolution
                        shape = (filter_num, img_size_h_cov, img_size_w_cov)
    pool_size -- 2 by default
    stride ----- 2 by default
    
    Returns:
    pool_result -- image pile after average pooling,
                   shape = (filter_num, img_size_h_pool, img_size_w_pool)
    '''
    
    filter_num = feature_map_pile.shape[0]
    
    img_size_h_cov = feature_map_pile.shape[1]
    img_size_w_cov = feature_map_pile.shape[2]
    
    img_size_h_pool = img_size_h_cov // pool_size
    img_size_w_pool = img_size_w_cov // pool_size
    
    pool_result = np.zeros((filter_num,
                            img_size_h_pool,
                            img_size_w_pool))
    
    
    for img_index in range(filter_num):
        r2 = 0
        for m in range(0, pool_size*(img_size_h_pool)-1, stride):
            c2 = 0
            for n in range(0, pool_size*(img_size_w_pool)-1, stride):
                pool_result[img_index,r2,c2] = np.average(feature_map_pile[img_index,m:m+pool_size,n:n+pool_size])
                c2 += 1
            r2 += 1
    return pool_result
    
def relu(feature_maps):
    relu_out = np.zeros(feature_maps.shape)
    for map_num in np.arange(feature_maps.shape[-1]):
        for m in np.arange(feature_maps.shape[0]):
            for n in np.arange(feature_maps.shape[1]):
#                relu_out[map_num,m,n] = np.max(0,feature_maps[map_num,m,n])
                '''effective, althought not efficient'''
                if feature_maps[m,n,map_num] < 0:
                    relu_out[m,n,map_num] = 0
                else:
                    relu_out[m,n,map_num] = feature_maps[m,n,map_num]
    return relu_out
        
def sigmoid(x):
    return 1. / (1 + np.exp(x))

    
    
    
    