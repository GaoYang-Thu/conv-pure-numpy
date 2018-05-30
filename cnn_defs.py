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


def cov2d(img, conv_filter):
    filter_size = conv_filter.shape[1]
    result = np.zeros((img.shape))
    for m in np.arange(img.shape[0] - filter_size +1):
        for n in np.arange(img.shape[1] - filter_size +1):
            curr_region = img[m:m + filter_size,
                              n:n + filter_size]
            '''remember to flip the kernel for convolution'''
            conv_filter_fliped = np.flipud(np.fliplr(conv_filter))
            curr_result = curr_region * conv_filter_fliped
            conv_sum = np.sum(curr_result)
            result[m,n] = conv_sum
    final_result = result[:img.shape[0] - filter_size + 1,
                          :img.shape[1] - filter_size + 1]
    return final_result
    '''
    scipy offers 2d method: signal.convolve2d
    I compared the results:
    r1 = cov2d(img, filter_1[0,:,:]) (my method)
    r3 = signal.convolve(img, filter_1[0,:,:],'valid') (scipy method)
    by using
    np.testing.assert_array_almost_equal(r1,r3)
    no errors were given, thus r1 and r3 are almost the same
    r1 - r3 is not 0, but the difference is fairly small to pass the 'almost same' array test
    '''

def conv(img, conv_filter):
    if len(img.shape) > 2 or len(conv_filter.shape) > 3:
        if img.shape[-1] != conv_filter.shape[-1]:
            print('Error: Number of channels in image and filter must match')
            sys.exit()
            
    if conv_filter.shape[1] != conv_filter.shape[2]:
        print('Error: Filter must be square. Row numbers and columns numbers must match.')
        sys.exit()
    
    if conv_filter.shape[1] % 2 == 0:
        print('Error: Filter size must be odd.')
        sys.exit()
    
    feature_maps = np.zeros((img.shape[0] - conv_filter.shape[1] +1,
                            img.shape[1] - conv_filter.shape[2] +1,
                            conv_filter.shape[0]))
    
    for filter_index in np.arange(conv_filter.shape[0]):
        print('Using filter number {}...'.format(filter_index + 1))
        curr_filter = conv_filter[filter_index,:,:]
        
        if len(curr_filter.shape) > 2:
            conv_map = cov2d(img[:,:,0],curr_filter[:,:,0])
            for channel_num in np.arange(curr_filter.shape[-1]):
                conv_map += cov2d(img[:,:,channel_num],curr_filter[:,:,channel_num])
        else:
            conv_map = cov2d(img,curr_filter)
            
        feature_maps[:,:,filter_index] = conv_map
    
    return feature_maps
    
def pooling(feature_maps, pool_size = cf.POOLING_SIZE, stride = cf.POOLING_STRIDE):
    pool_out = np.zeros((feature_maps.shape[0] // pool_size,
                         feature_maps.shape[1] // pool_size,
                         feature_maps.shape[-1],))
    
    for map_num in np.arange(feature_maps.shape[-1]):
        r2 = 0
        for m in np.arange(0, 2*(feature_maps.shape[0]//pool_size)-1, stride):
            c2 = 0
            for n in np.arange(0, 2*(feature_maps.shape[1]//pool_size)-1, stride):
                pool_out[r2,c2,map_num] = np.max(feature_maps[m:m+pool_size, n:n+pool_size, map_num])
                c2 += 1
            r2 += 1
    return pool_out
    
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
        
    
    
    
    
    
    