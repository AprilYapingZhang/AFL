#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, All rights reserved.
# Author:  Yaping Zhang, Shan Liang, Shuai Nie, Wenju Liu, Shouye Peng
# Title: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data;
# Journal: Pattern Recognition Letters 2018

# tensorflow-gpu == 1.10.1
# python version > 3.6


from __future__ import print_function

import argparse
import os

import numpy as np

from model import AFLModel
from utils import config_tensorlfow
from utils import load_h5py_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_weights', default='./pre_weights.hdf5',
                        help='model directory, which should be writable')
    parser.add_argument('--data_dir',
                        default='./data',
                        help='training data directory')
    parser.add_argument('--datasets', default='CASIA_HWDB_1.0_1.1_data',
                        help='subdir of training data to use, separated by comma')
    parser.add_argument('--output_dir', default='./weights',
                        help='model directory, which should be writable')
    parser.add_argument('--adam_lr', type=np.float32, default=2e-4,
                        help='learning rate')
    parser.add_argument('--adam_beta_1', type=np.float32, default=0.5,
                        help='adam beta_1')
    parser.add_argument('--img_w', type=int, default=64,
                        help='normalized width of the input image to network')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='size of one training batch')
    parser.add_argument('--alpha', type=int, default=0.1, help='alpha value in AFL')
    parser.add_argument('--char_num', type=int, default=3756, help='number of class')

    opt = parser.parse_args()

    config_tensorlfow()
    np.random.seed(1337)

    weights_output_dir = opt.output_dir
    if not os.path.exists(weights_output_dir):
        os.makedirs(weights_output_dir)

    # data preparing
    CASIA_file_trn = os.path.join(opt.data_dir,
                                  'CASIA_HWDB_1.0_1.1_data/trn-HWDB1.0-1.1-3756-uint8.hdf5')
    CASIA_file_tst = os.path.join(opt.data_dir,
                                  "CASIA_HWDB_1.0_1.1_data/tst-HWDB1.0-1.1-3756-uint8.hdf5")
    printed_file = os.path.join(opt.data_dir, 'CASIA_HWDB_1.0_1.1_data/norm_hand_pair_3755.hdf5')

    x_train_handwritten, y_train_handwritten = load_h5py_data(CASIA_file_trn)
    x_train_printed, y_train_printed = load_h5py_data(printed_file)
    X_test, y_test = load_h5py_data(CASIA_file_tst)

    afl_model = AFLModel(opt)
    afl_model.training(x_train_handwritten, y_train_handwritten,
                       x_train_printed, y_train_printed,
                       X_test, y_test)


if __name__ == '__main__':
    main()
