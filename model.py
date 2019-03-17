#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, All rights reserved.
# Author:  Yaping Zhang, Shan Liang, Shuai Nie, Wenju Liu, Shouye Peng
# Title: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data;
# Journal: Pattern Recognition Letters 2018


from __future__ import print_function

import os

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from six.moves import range

from module import build_domain_classifier
from module import build_feature_extractor
from module import build_label_predictor


class AFLModel(object):
    """
     build AFL model
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
        self.alpha = opt.alpha

        #  Basic parameters of training models
        self.input_shape = (self.img_w, self.img_w, 1)
        self.optimizer = Adam(lr=self.adam_lr, beta_1=self.adam_beta_1)
        self.batch_size = opt.batch_size
        self.nb_epochs = opt.num_epochs

        # build model
        self._build()
        self.pretrain_weights = opt.pretrain_weights
        self._initialize_weights(self.pretrain_weights)

    def _build_feature_extractor(self):
        # build the feature_extractor
        self.feature_extractor = build_feature_extractor(self.input_shape)
        self.feature_extractor.compile(
            optimizer=self.optimizer,
            loss=['mean_squared_error']
        )

    def _build_label_predictor(self):
        # build the label_predictor
        self.label_predictor = build_label_predictor(self.char_num)
        self.label_predictor.compile(
            optimizer=self.optimizer,
            metrics=['accuracy'],
            loss=['sparse_categorical_crossentropy']
        )

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

    def _build_domain_classifier(self):
        # build the domain_classifier
        self.domain_classifier = build_domain_classifier()
        self.domain_classifier.compile(
            optimizer=self.optimizer,
            loss=['binary_crossentropy'])

    def _build_domain_adaptation_combined(self):
        # build the domain_adaptation_combined
        self.label_predictor.trainable = False
        self.domain_classifier.trainable = False
        img_hw = Input(shape=self.input_shape)
        img_pr = Input(shape=self.input_shape)
        share_feature_hw = self.feature_extractor([img_hw])
        share_feature_pr = self.feature_extractor([img_pr])
        trick_hw = self.domain_classifier(share_feature_hw)
        trick_pr = self.domain_classifier(share_feature_pr)
        y_label_hw = self.label_predictor(share_feature_hw)
        y_label_pr = self.label_predictor(share_feature_pr)

        self.domain_adaptation_combined = Model(inputs=[img_hw, img_pr],
                                                outputs=[trick_hw, trick_pr, y_label_hw,
                                                         y_label_pr])
        self.domain_adaptation_combined.compile(
            optimizer=self.optimizer,
            metrics=['accuracy'],
            loss=['binary_crossentropy', 'binary_crossentropy',
                  'sparse_categorical_crossentropy',
                  'sparse_categorical_crossentropy'],
            loss_weights=[self.alpha, self.alpha, 1, 1]
        )
        print(self.domain_adaptation_combined.summary())

    def _build(self):
        self._build_feature_extractor()
        self._build_label_predictor()
        self._build_character_classifier()
        self._build_domain_classifier()
        self._build_domain_adaptation_combined()

    def _initialize_weights(self, pretrain_weights):
        self.character_classifier.load_weights(pretrain_weights, by_name=True)

    def training(self,
                 x_train_hw, y_train_hw,
                 x_train_pr, y_train_pr,
                 x_test, y_test):

        """Alternatively training models"""
        # get a batch of real images
        nb_train_hw, nb_train_pr = x_train_hw.shape[0], x_train_pr.shape[0]
        num_truncate = nb_train_hw % self.batch_size
        hw_data_used_num = nb_train_hw - num_truncate

        # get a batch of real images
        num_truncate = nb_train_pr % self.batch_size
        pr_data_used_num = nb_train_pr - num_truncate
        score = self.test(x_test, y_test)
        print('\nTesting for pretraining weights: accuracy %04f' % (score[1]))
        for epoch in range(self.nb_epochs):
            print('Epoch {} of {}'.format(epoch + 1, self.nb_epochs))
            nb_batches = int(nb_train_hw / self.batch_size)
            progress_bar = Progbar(target=nb_batches)

            epoch_label_predictor_loss = []
            epoch_domain_classifier_loss = []
            epoch_domain_adaptation_combined_loss = []
            for index in range(nb_batches):
                progress_bar.update(index)

                # get a batch of handwritten data
                hw_data_index_start = index * self.batch_size % hw_data_used_num
                hw_data_index_end = hw_data_index_start + self.batch_size
                img_hw = x_train_hw[hw_data_index_start:hw_data_index_end]
                cls_labels_hw = y_train_hw[hw_data_index_start:hw_data_index_end]
                domain_labels_hw = np.array([0] * self.batch_size)

                # get a batch of printed data
                pr_data_index_start = index * self.batch_size % pr_data_used_num
                pr_data_index_end = pr_data_index_start + self.batch_size
                img_pr = x_train_pr[pr_data_index_start:pr_data_index_end]
                cls_labels_pr = y_train_pr[pr_data_index_start:pr_data_index_end]
                domain_labels_pr = np.array([1] * self.batch_size)

                share_feature_hw = self.feature_extractor.predict([img_hw])
                share_feature_pr = self.feature_extractor.predict([img_pr])

                # alternativily training,combined data
                share_feature_hp = np.concatenate([share_feature_hw, share_feature_pr], axis=0)
                domain_labels_hp = np.concatenate([domain_labels_hw, domain_labels_pr], axis=0)

                # updating parameters of domain_classifier
                epoch_domain_classifier_loss.append(
                    self.domain_classifier.train_on_batch(share_feature_hp, domain_labels_hp))

                # updating parameters of feature_extractor
                trick_hw = np.array([1] * self.batch_size)
                trick_pr = np.array([0] * self.batch_size)
                epoch_domain_adaptation_combined_loss.append(
                    self.domain_adaptation_combined.train_on_batch(
                        [img_hw, img_pr],
                        [trick_hw, trick_pr, cls_labels_hw, cls_labels_pr]))

                # updating parameters of label_predictor
                share_feature_hw = self.feature_extractor.predict([img_hw])
                epoch_label_predictor_loss.append(
                    self.label_predictor.train_on_batch([share_feature_hw], [cls_labels_hw]))

            score = self.test(x_test, y_test)
            weights_output_dir = os.path.join(self.output_dir,
                                              'weights%02d-%04f.h5' % (
                                                  epoch, score[1]))
            self.save_weights(weights_output_dir)
            print('\nTesting for epoch %02d: accuracy %04f' % (epoch + 1, score[1]))

    def test(self, x_test, y_test):
        score = self.character_classifier.evaluate(x_test, y_test, verbose=False)
        return score

    def save_weights(self, weights_output_dir):
        self.character_classifier.save_weights(weights_output_dir)
