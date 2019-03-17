#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018, All rights reserved.
# Author:  Yaping Zhang, Shan Liang, Shuai Nie, Wenju Liu, Shouye Peng
# Title: Robust offline handwritten character recognition through exploring writer-independent features under the guidance of printed data;
# Journal: Pattern Recognition Letters 2018

from __future__ import print_function

from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model

from normalization import BatchNormalization


def build_feature_extractor(input_shape):
    # build a feature extrator
    input_img = Input(shape=input_shape)

    conv1_gen = Conv2D(96, (3, 3), strides=(1, 1), padding='same',
                       kernel_initializer='glorot_normal',
                       name='conv1_gen')(input_img)
    BN1_gen = BatchNormalization()(conv1_gen)
    LeakyReLU1_gen = LeakyReLU(name='LeakyReLU1_gen')(BN1_gen)
    conv2_gen = Conv2D(96, (3, 3), padding='same', strides=(2, 2),
                       kernel_initializer='glorot_normal',
                       name='conv2_gen')(LeakyReLU1_gen)
    BN2_gen = BatchNormalization()(conv2_gen)
    LeakyReLU2_gen = LeakyReLU(name='LeakyReLU2_gen')(BN2_gen)

    conv3_gen = Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal',
                       name='conv3_gen')(
        LeakyReLU2_gen)
    BN3_gen = BatchNormalization()(conv3_gen)
    LeakyReLU3_gen = LeakyReLU(name='LeakyReLU3_gen')(BN3_gen)

    conv4_gen = Conv2D(128, (3, 3), padding='same', strides=(2, 2),
                       kernel_initializer='glorot_normal',
                       name='conv4_gen')(LeakyReLU3_gen)
    BN4_gen = BatchNormalization()(conv4_gen)
    LeakyReLU4_gen = LeakyReLU(name='LeakyReLU4_gen')(BN4_gen)

    conv5_gen = Conv2D(160, (3, 3), padding='same', kernel_initializer='glorot_normal',
                       name='conv5_gen')(
        LeakyReLU4_gen)
    BN5_gen = BatchNormalization()(conv5_gen)
    LeakyReLU5_gen = LeakyReLU(name='LeakyReLU5_gen')(BN5_gen)
    conv6_gen = Conv2D(160, (3, 3), padding='same', strides=(2, 2),
                       kernel_initializer='glorot_normal',
                       name='conv6_gen')(LeakyReLU5_gen)
    BN6_gen = BatchNormalization()(conv6_gen)
    LeakyReLU6_gen = LeakyReLU(name='LeakyReLU6_gen')(BN6_gen)

    conv7_gen = Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal',
                       name='conv7_gen')(
        LeakyReLU6_gen)
    BN7_gen = BatchNormalization()(conv7_gen)
    LeakyReLU7_gen = LeakyReLU(name='LeakyReLU7_gen')(BN7_gen)

    conv8_gen = Conv2D(256, (3, 3), padding='same', strides=(2, 2),
                       kernel_initializer='glorot_normal',
                       name='conv8_gen')(LeakyReLU7_gen)
    BN8_gen = BatchNormalization()(conv8_gen)
    LeakyReLU8_gen = LeakyReLU(name='LeakyReLU8_gen')(BN8_gen)

    conv9_gen = Conv2D(256, (3, 3), kernel_initializer='glorot_normal', name='conv9_gen')(
        LeakyReLU8_gen)
    BN9_gen = BatchNormalization()(conv9_gen)
    LeakyReLU9_gen = LeakyReLU(name='LeakyReLU9_gen')(BN9_gen)

    feature = Flatten(name='cls_flatten')(LeakyReLU9_gen)

    return Model(inputs=[input_img], outputs=feature, name='feature_extractor_model')


def build_label_predictor(char_num):
    # build a label predictor
    feature = Input(shape=(1024,))
    dense1 = Dense(512, activation='relu', name='dense1')(feature)
    dense1 = Dropout(0.5)(dense1)
    label_pred = Dense(char_num, activation='softmax', name='label_pred_3756')(dense1)
    return Model(inputs=[feature], outputs=label_pred, name='Label_Predictor_model')


def build_domain_classifier():
    # build a domain classifier
    feature = Input(shape=(1024,))
    dense1 = Dense(512, activation='relu', name='dense1_d')(feature)
    dense1 = Dropout(0.5)(dense1)
    Domain_label = Dense(1, activation='sigmoid', name='Domain_label')(dense1)
    return Model(inputs=feature, outputs=[Domain_label], name='Domain_classifier')
