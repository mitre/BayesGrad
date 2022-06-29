import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad-pet')[0] + 'xnn4rad-pet/'
sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np
from codebase.vanilla_gradients import VanillaGradients

"""

Implementation of Smoothgrad and Smoothgrad derivatives (i.e. Smoothgrad^2,
VarGrad). This package expects that the model's 
last layer is logits with no activation function applied.


Core Module for SmoothGrad Algorithm
Primarliy adapted from tf-explain
https://github.com/sicara/tf-explain


Parts of code adapted from Saliency package
https://github.com/PAIR-code/saliency


# Copyright 2021 Google Inc. All Rights Reserved.
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
"""

class SmoothGrad:
    def __init__(self):
        self.base_explainer = VanillaGradients()

    """
    Perform SmoothGrad algorithm for a given input
    Paper: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
    """

    def explain(self, images, model, class_index, num_samples=5, stdev_spread=0.15, uncertainty=None, squared=False, dropout=False):
        """
        Compute SmoothGrad for a specific class index

        Args:
            images : tf.tensor
                tensor of images to perform the explainer method on
                Shape is (num_images, H, W, C)
            model : tf.keras.Model
                tf.keras model to inspect
            class_index : int
                Index of targeted class
            num_samples : int
                Number of noisy samples to average in SmoothGrad
            stdev_spread : float
                Amount of noise to add to the input, as fraction of the
                total spread (x_max - x_min). Defaults to 15%. 
            uncertainty : str
                Optional argument to take variance or standard deviation of
                noisy examples rather than average. Valid options are 'std',
                'var', and None. If uncertainty is None, than function
                returns the mean of the noisy gradients.
            squared : bool
                Whether to square gradients
            dropout : bool
                If model has been trained as BNN according to dropout as 
                Bayesian approximation, option ot leave dropout on at inference
                for each example when taking gradients. 
                
        Returns:
            np.array of smooth grad gratients. Shape matches input shape
            (num_images, H, W, C).
        """
        stdev = stdev_spread * (np.max(images[0]) - np.min(images[0])) 

        noisy_images = self.generate_noisy_images(images, num_samples, stdev)

        s_maps = []
        for i, noisy_image in enumerate(noisy_images):
            s_map = self.base_explainer.explain(np.array([noisy_image]), model, class_index, deterministic=not dropout, squared=squared)
            s_maps.append(s_map)

        s_maps = np.array(s_maps)

        if uncertainty:
            if uncertainty == 'std':
                std = np.abs(s_maps.std(axis=0))
                return std
            elif uncertainty == 'var':
                var = s_maps.var(axis=0)
                return var
            else:
                raise ValueError("Uncertainty must be 'std' or 'var' or None")
        else:
            s_maps_mean = s_maps.mean(axis=0)
            return s_maps_mean

    @staticmethod
    def generate_noisy_images(images, num_samples, noise):
        """
        Generate num_samples noisy images with std noise for each image.
        Args:
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution
        Returns:
            np.ndarray: 4D-Tensor of noisy images with shape (batch_size*num_samples, H, W, 3)
        """
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)
        return repeated_images + noise

