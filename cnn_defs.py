# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:23:01 2018

@author: YangGao
"""

import numpy as np
from scipy import signal
import config_cov as cf


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

#    cache_single = (img_single, filter_single)
#    return result_single, cache_single
    return result_single
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






def conv_single_pile(img_single, filter_pile):
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

        # print('Using filter number: {}...'.format(filter_index + 1))

        curr_filter = filter_pile[filter_index,:,:]

        feature_map_single = conv2d(img_single, curr_filter) # we want only the conv result

        feature_map_pile[filter_index,:,:] = feature_map_single

    # print('convolution using {} filters is completed'.format(filter_num))
    return feature_map_pile






def conv_pile_group(img_pile, filter_group):
    '''
    2d convolution of a pile of gray images and a group of piles of convolution filters

    Arguments:
        img_pile ----- a pile of images to conv, 3d array
                       shape = (img_num, img_size_h, img_size_w)
                       filter_group -- a group of piles of filters, 4d array
                       shape = (pile_num, filter_num, filter_size, filter_size)

    Returns:
        feature_map_pile -- a plie of convolution results,
                            shape = (filter_num, img_size_h_cov, img_size_w_cov)

    '''

    (pile_num, filter_num, filter_size, filter_size) = filter_group.shape

    (img_num, img_size_h, img_size_w) = img_pile.shape

    img_size_h_cov = img_size_h - filter_size + 1
    img_size_w_cov = img_size_w - filter_size + 1

    feature_map_pile = np.zeros((filter_num,
                                 img_size_h_cov,
                                 img_size_w_cov))

    for img_index in range(img_num):
        # print('computing image number: {}...'.format(img_index+1))

        # the single image for convolution
        img_single = img_pile[img_index,:,:]

        # the filter pile for convolution
        filter_pile = filter_group[img_index,:,:,:]

        # recursively add convolution results
        feature_map_pile += conv_single_pile(img_single, filter_pile)

        # print('computing image number: {} is completed.\n'.format(img_index+1))

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
    # print('average pooling is done.')
    return pool_result






def relu(feature_map_pile):
    '''
    relu: y = x (if x > 0) or =0 (if x<=0)

    Arguments:
        feature_map_pile -- image pile after convolution
                        shape = (filter_num, img_size_h_cov, img_size_w_cov)


    Returns:
        relu_result -- image pile after relu,
                   shape = (filter_num, img_size_h_pool, img_size_w_pool)
    '''
    relu_result = feature_map_pile

    for i in np.ndindex(feature_map_pile.shape):
                #  effective, althought not efficient
                # if x<0, replace x with 0
                if feature_map_pile[i] < 0:
                    relu_result[i] = 0
    # print('relu activation is completed.')
    return relu_result






def sigmoid(x):
    return 1. / (1 + np.exp(-x))






def vectorize_column(img_single):
    '''
    vectorize a single img to an array in column wise

    Arguments:
        img_single -- a 2d array

    Returns:
        v_result -- an 1d array
    '''

    v_result = np.concatenate(img_single.T)
    return v_result





def vector_and_concat(img_pile):
    '''
    1,vectorize every img in the pile into an 1d array
    2,concatenate those arrays into a long array

    Arguments:
        img_pile -- an image pile after convolution,
                shape = (img_num, img_size_h_cov, img_size_w_cov)

    Intermediate result:
        img_array -- a 2d array, shape = (img_num, img_size_h_cov * img_size_w_cov)

    Returns:
        vc_result -- an 1d array,
                 length = (filter_num * img_size_h_pool * img_size_w_pool)

    '''

    (img_num, img_size_h_cov, img_size_w_cov) = img_pile.shape

    # create a 2d array: img_array
    # then store vectorized result of each image (= an 1d array) in each column of img_array
    img_array_2d = np.zeros((img_size_h_cov * img_size_w_cov, img_num))

    for img_index in range(img_num):
        img_array_2d[:,img_index] = vectorize_column(img_pile[img_index,:,:])

    vc_result = vectorize_column(img_array_2d)

    return vc_result






def calculate_loss(true_label, output_label):
    '''
    calculate loss between true label and output label

    Arguments:
        true label: 2d column vector
        output label: 2d column vector

    Returns:
        loss_num: a scalar
    '''

    if len(true_label) != len(output_label):
        print('true label and output label must match.')

    error_array = true_label - output_label
    loss_num = 0.5 * np.dot(error_array.T, error_array)
    return loss_num






def generate_row_vector(array_1d):
    '''
    generate an 2d row vector from 1d array

    Arguments:
        array_1d --------   shape = (length,)


    Returns:
        row_vector --       shape = (1, length)
    '''
    array_length = len(array_1d)
    row_vector = array_1d

    row_vector = row_vector.reshape(1, array_length)
    return row_vector





def generate_column_vector(array_1d):
    '''
    generate an 2d column vector from 1d array

    Arguments:
        array_1d --------   shape = (length,)


    Returns:
        column_vector --       shape = (length, 1)
    '''
    array_length = len(array_1d)
    column_vector = array_1d

    column_vector = column_vector.reshape(array_length, 1)
    return column_vector






def reverse_vc(column_vector):
    '''
    1,split the column vector into sub vectors
    2,regroup each subvector into a 2d array

    Arguments:
        column_vector -- a column vector
                 shape = (img_num * img_size_h_cov * img_size_w_cov)

    Intermediate result:
        img_array -- a 2d array pile, each img is transposed
                 shape = (img_num, img_size_w_cov * img_size_h_cov)

    Returns:
        reverse_vc_result -- a pile of 2d arrays,
                 shape = (img_num, img_size_h_cov, img_size_w_cov)

    '''
    img_num = cf.L2_FILTER_NUM
#    img_size = 8
    img_size = cf.L2_FILTER_OUTPUT_SIZE_AFTER_POOLING

    img_array = column_vector.reshape(img_num, img_size, img_size)
    reverse_vc_result = np.zeros(img_array.shape)

    for img_index in range(img_array.shape[0]):
        reverse_vc_result[img_index,:,:] = img_array[img_index,:,:].T

#    print('reverse v&c is done.')
    return reverse_vc_result






def upsampling(img_pile):
    '''
    unsample the img_pile, ratio = 2 in each axis

    Arguments:
        img_pile --------   shape = (img_num, size_before_pooling, size_before_pooling)


    Returns:
        img_pile_up -----   upsampled img pile.
                        shape = (img_num, size_before_pooling * 2, size_before_pooling * 2)
    '''


    (img_num,size_before_pooling,size_before_pooling) = img_pile.shape

    size_after_pooling = int(size_before_pooling * 2)

    img_pile_up_shape = (img_num, size_after_pooling, size_after_pooling)

    img_pile_up = np.zeros(img_pile_up_shape)

    for img_index in range(img_num):
        img_single_up = img_pile_up[img_index,:,:]
        img_single_pooled = img_pile[img_index,:,:]
        for i in np.ndindex(img_single_up.shape):
            index_pooled = (int(np.floor(0.5*i[0])), int(np.floor(0.5*i[1])))
#            print(i, index_pooled)
            img_single_up[i] = 0.25 * img_single_pooled[index_pooled]


    return img_pile_up





#   ---------------------------------------------------------------
#
#   CALCULATE BACKPROBAGATION USING MATRIX MULTIPLICATION INSTEAD
#
#   ---------------------------------------------------------------

#def conv2d_backward(d_result_single, cache_single):
#    '''
#    backward of 2d convolution of a single image and a single filter
#
#    Arguments:
#    d_result_single -- gradient of the loss with respect to the output of conv2d
#                       shape = (img_size_h_cov, img_size_w_cov)
#    cache ------------ output of conv2d()
#
#    Returns:
#    d_img_single ----- gradient of the loss with respect to the input of conv2d,
#                    shape = (img_size_h, img_size_w)
#    d_filter_single -- gradient of the loss with respect to the filter of conv2d,
#                    shape = (filter_size, filter_size)
#    '''
#
#    (img_single, filter_single) = cache_single
#
#    (img_size_h, img_size_w) = img_single.shape
#    (filter_size, filter_size) = filter_single.shape
#    (img_size_h_cov, img_size_w_cov) = d_result_single.shape
#
#    d_img_single = np.zeros(img_single.shape)
#    d_filter_single = np.zeros(filter_single.shape)
#
#    for h in range(img_size_h_cov):
#        for w in range(img_size_w_cov):
#
#            d_img_single[h:h+filter_size, w:w+filter_size] += filter_single * d_result_single[h,w]
#
#            d_filter_single_raw = img_single[h:h+filter_size, w:w+filter_size] * d_result_single[h,w]
#            # flip back
#            d_filter_single += np.fliplr(np.flipud(d_filter_single_raw))
#
#    return d_img_single, d_filter_single






def paddle_zeros(img_single):
    '''
    extend img_single array with (paddle_length) rows and columns of zeros

    Arguments:
        img_single -- 2d array, shape = (img_size,img_size)

    Returns:
        img_single_paddled -- 2d array, shape = (img_size + 2*paddle_length, img_size + 2*paddle_size)
    '''

    paddle_length = cf.L2_FILTER_SIZE - 1

    start_index = paddle_length

    img_size = img_single.shape[0]

    img_size_new = img_size + 2 * paddle_length

    img_single_paddled = np.zeros((img_size_new,img_size_new))

    img_single_paddled[start_index:start_index + img_size,
                       start_index:start_index + img_size] = img_single

    return img_single_paddled







def d_fc_weights():
    pass

def d_fc_thresholds():
    pass

def d_l2_filter_group():
    pass

def d_l2_thresholds():
    pass

def d_l1_filter_pile():
    pass

def d_l1_thresholds():
    pass


def cnn_forward(img):
    '''
    pass img through the cnn forwardly

    Arguments:
        img -- 2d array

    Returns:
        loss -- loss between true label of the img and calculate label

    '''
    pass

def cnn_backward():
    pass

