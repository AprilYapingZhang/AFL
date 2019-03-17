#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, All rights reserved.
# Author:  Yaping Zhang, Shan Liang, Shuai Nie, Wenju Liu, Shouye Peng
# Title: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data;
# Journal: Pattern Recognition Letters 2018


from __future__ import print_function

import h5py
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def config_tensorlfow():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))


def load_h5py_data(data_file):
    with h5py.File(data_file, 'r') as f:
        X = f['data/x'][:]
        y = f['data/t'][:].reshape(-1, )
    return X, y

