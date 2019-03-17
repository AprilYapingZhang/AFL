#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, All rights reserved.
# Author:  Yaping Zhang, Shan Liang, Shuai Nie, Wenju Liu, Shouye Peng
# Title: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data;
# Journal: Pattern Recognition Letters 2018


from __future__ import print_function

import argparse
import os

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from six.moves import range

from module import build_feature_extractor, build_label_predictor
from utils import config_tensorlfow
from utils import load_h5py_data


class BaselineModel(object):
    """
     build a character classifier model
     The architecture is same as the AFL backbone.
     Args:
         opt: model config options
    """

    def __init__(self, opt):
        self.opt = opt
        self.output_dir = opt.output_dir
        self.img_w = opt.img_w
        self.char_num = opt.char_num
        self.adam_lr = opt.adam_lr
        self.adam_beta_1 = opt.adam_beta_1

        #  Basic parameters of training models
        self.input_shape = (self.img_w, self.img_w, 1)
        self.optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1)
        self.batch_size = opt.batch_size
        self.nb_epochs = opt.num_epochs

        # build model
        self._build()

    def _build_feature_extractor(self):
        # build the feature_extractor
        self.feature_extractor = build_feature_extractor(self.input_shape)

    def _build_label_predictor(self):
        # build the label_predictor
        self.label_predictor = build_label_predictor(self.char_num)

    def _build_character_classifier(self):
        # build the character_classifier
        input_img = Input(shape=self.input_shape)
        # get a new representation
        self.share_feature = self.feature_extractor([input_img])
        self.y_label = self.label_predictor(self.share_feature)

        self.character_classifier = Model(inputs=[input_img], outputs=[self.y_label])
        self.character_classifier.compile(
            optimizer=self.optimizer,
            metrics=['accuracy'],
            loss='sparse_categorical_crossentropy')
        print(self.character_classifier.summary())

    def _build(self):
        self._build_feature_extractor()
        self._build_label_predictor()
        self._build_character_classifier()

    def training(self,
                 x_train_hw, y_train_hw,
                 x_test, y_test):

        """Alternatively training models"""
        # get a batch of real images
        nb_train_hw = x_train_hw.shape[0]
        num_truncate = nb_train_hw % self.batch_size
        hw_data_used_num = nb_train_hw - num_truncate

        for epoch in range(self.nb_epochs):
            print('Epoch {} of {}'.format(epoch + 1, self.nb_epochs))
            nb_batches = int(nb_train_hw / self.batch_size)
            progress_bar = Progbar(target=nb_batches)

            epoch_label_predictor_loss = []

            for index in range(nb_batches):
                progress_bar.update(index)

                # get a batch of handwritten data
                hw_data_index_start = index * self.batch_size % hw_data_used_num
                hw_data_index_end = hw_data_index_start + self.batch_size
                img_hw = x_train_hw[hw_data_index_start:hw_data_index_end]
                cls_labels_hw = y_train_hw[hw_data_index_start:hw_data_index_end]

                # updating parameters of label_predictor
                epoch_label_predictor_loss.append(
                    self.character_classifier.train_on_batch([img_hw], [cls_labels_hw]))

            score = self.test(x_test, y_test)
            weights_output_dir = os.path.join(self.output_dir,
                                              'pre_weights%02d-%04f.h5' % (
                                                  epoch, score[1]))
            self.save_weights(weights_output_dir)
            print('\nTesting for epoch %02d: accuracy %04f' % (epoch + 1, score[1]))

    def test(self, x_test, y_test):
        score = self.character_classifier.evaluate(x_test, y_test, verbose=False)
        return score

    def save_weights(self, weights_output_dir):
        self.character_classifier.save_weights(weights_output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='./data',
                        help='training data directory')
    parser.add_argument('--datasets', default='CASIA_HWDB_1.0_1.1_data',
                        help='subdir of training data to use, separated by comma')
    parser.add_argument('--output_dir', default='./pre_weights',
                        help='model directory, which should be writable')
    parser.add_argument('--adam_lr', type=np.float32, default=2e-4,
                        help='learning rate')
    parser.add_argument('--adam_beta_1', type=np.float32, default=0.5,
                        help='adam beta_1')
    parser.add_argument('--img_w', type=int, default=64,
                        help='normalized width of the input image to network')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='size of one training batch')
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

    x_train_handwritten, y_train_handwritten = load_h5py_data(CASIA_file_trn)
    x_test, y_test = load_h5py_data(CASIA_file_tst)

    baseline_model = BaselineModel(opt)
    baseline_model.training(x_train_handwritten, y_train_handwritten,
                            x_test, y_test)


if __name__ == '__main__':
    main()
