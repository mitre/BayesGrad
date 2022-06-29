# From https://github.com/OATML/bdl-benchmarks/blob/alpha/baselines/diabetic_retinopathy_diagnosis/mc_dropout/model.py
# This file has been modified
#
#
# Copyright 2019 BDL Benchmarks Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definition of the VGGish network for Monte Carlo Dropout baseline."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad')[0] + 'xnn4rad/'
sys.path.append(BASE_DIR)
import numpy as np
import pandas as pd
import tensorflow as tf
tfk = tf.keras
tfkl = tfk.layers
tfkm = tfk.metrics
from codebase.vanilla_gradients import VanillaGradients
import matplotlib.pyplot as plt
import seaborn as sns


def VGGDrop(dropout_rate, num_base_filters, learning_rate, l2_reg, input_shape, num_classes=2, filterwise_dropout=True):
  """VGG-like model with dropout.
  Args:
    dropout_rate: `float`, the rate of dropout, between [0.0, 1.0).
    num_base_filters: `int`, number of convolution filters in the
      first layer.
    learning_rate: `float`, ADAM optimizer learning rate.
    l2_reg: `float`, the L2-regularization coefficient.
    input_shape: `iterable`, the shape of the images in the input layer.
  Returns:
    A tensorflow.keras.Sequential VGG-like model with dropout.
  """
  filterwise_dropout = filterwise_dropout

  # set noise shape for dropout layers
  if filterwise_dropout:
    noise_shape = [None, 1, 1, None]
  else:
    noise_shape = None

  if num_classes == 2:
    last_layer_units = 1
  elif num_classes > 2:
    last_layer_units = num_classes

  # Feedforward neural network
  model = tfk.Sequential([
      tfkl.InputLayer(input_shape),
      # Block 1 [(conv, bn, relu, dropout) --> mp]
      tfkl.Conv2D(filters=num_base_filters,
                  kernel_size=(3,3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block1_conv1'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters,
                  kernel_size=(3,3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block1_conv2'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.MaxPooling2D(pool_size=(2,2), name='max_pool_1'),

      # Block 2 [(conf, bn, relu, dropout) x2 --> mp] 
      tfkl.Conv2D(filters=num_base_filters * 2,
                  kernel_size=(3,3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block2_conv1'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 2,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block2_conv2'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.MaxPooling2D(pool_size=(2, 2), name='max_pool_2'),

      # Block 3 [(conv, bn, relu, dropout) x2 --> mp]
      tfkl.Conv2D(filters=num_base_filters * 4,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block3_conv1'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 4,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block3_conv2'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.MaxPooling2D(pool_size=(2, 2), name='max_pool_3'),

      # Block 4 [(conv, bn, relu, dropout) x 4 --> mp]
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block4_conv1'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block4_conv2'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block4_conv3'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block4_conv4'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='max_pool_4'),

      # Block 5 [(conv, bn, relu, dropout) x 3 --> (conv, bn, relu, global pooling, Dense, softmax)
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block5_conv1'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block5_conv2'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block5_conv3'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.Conv2D(filters=num_base_filters * 8,
                  kernel_size=(3, 3),
                  padding="same",
                  kernel_regularizer=tfk.regularizers.l2(l2_reg),
                  name='block5_conv4'),
      tfkl.BatchNormalization(),
      tfkl.Activation("relu"),
      tfkl.Dropout(dropout_rate, noise_shape=noise_shape),
      tfkl.MaxPooling2D(pool_size=(2, 2), data_format='channels_last', name='max_pool_5'),

      tfkl.Flatten(),
      # Output 
      tfkl.Dense(last_layer_units,
                 name='logits'),
  ])


  return model


