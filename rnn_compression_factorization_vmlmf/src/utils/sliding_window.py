################################################################################
# [VMLMF] Lowrank Matrix Factorization with Vector-Multiplication
# Project: Starlab
#
# Authors: Hyojin Jeon (tarahjjeon@snu.ac.kr), Seoul National University
#         U Kang (ukang@snu.ac.kr), Seoul National University
#
# File: sliding_window.py
# - utilities for process data using sliding window
#
# Version : 1.0
# Date : Oct 14, 2021
# Main Contact: Hyojin Jeon
#
# This software is free of charge under research purposes.
# For commercial purposes, please contact the authors.
#
################################################################################
# from http://www.johnvinyard.com/blog/?p=268
# pylint: disable=C0103, E1101, C0114, R0902,C0116, R0914, R0913, C0123, W0613, W0102,C0413, E0401
"""
====================================
 :mod:`sliding_window`
====================================
.. moduleauthor:: Hyojin Jeon  <tarahjjeon@snu.ac.kr>
설명
=====
sliding window를 사용하여 데이터를 전처리하기 위한 모듈입니다.

"""
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast

def norm_shape(shape):
    '''Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    :param shape: an int, or a tuple of ints
    :returns: a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a,ws,ss = None,flatten = True):
    '''Return a sliding window over a in any number of dimensions

    :param a: an n-dimensional numpy array
    :param ws: an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
    :param ss: an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
    :param flatten: if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
    :returns: an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
    print(shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        f'a.shape, ws and ss must all have the same length. They were {str(ls)}')

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        f'ws cannot be larger than a in any dimension.\
 a.shape was {str(a.shape)} and ws was {str(ws)}')

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides

    strided = ast(a,shape = newshape,strides = newstrides)

    if not flatten:
        return strided

    return np.squeeze(strided)
