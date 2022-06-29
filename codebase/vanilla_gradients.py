import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad-pet')[0] + 'xnn4rad-pet/'
sys.path.append(BASE_DIR)

import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import codebase.visualization_utils as vu

"""

Implementation of vanilla gradients. This package expects that the model's 
last layer is logits with no activation function applied.

Code adapted from tf-explain package hosted at:
https://github.com/sicara/tf-explain
"""

class VanillaGradients:
    """
    Perform gradients backpropagation for a given input
    Paper: [Deep Inside Convolutional Networks: Visualising Image Classification
        Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
    """

    def explain(self, images, model, class_index, deterministic=True, squared=False):
        """
        Perform gradients backpropagation for a given input

        Args:
            images : tf.tensor
                tensor of images to perform the explainer method on
                Shape is (num_images, H, W, C)
            model : tf.keras.Model
                tf.keras model to inspect
            class_index : int
                Index of targeted class
            deterministic : bool
                Whether to leave dropout on at prediction for calculation of
                gradients of class score with respect to input image for the 
                NN instantiated with weights from a single sample from 
                estimated Bayesian posterior (via dropout as bayesian 
                approximation)
            squared : bool
                Whether to square gradients
                
        Returns:
            np.array of gratients of logit at class_index with respect 
            to input images. Shape matches input (num_images, H, W, C).
        """
        gradients = VanillaGradients().compute_gradients(images, model, class_index, deterministic)
        gradients = gradients.numpy()
        if squared:
            gradients = gradients * gradients

        return gradients

    @staticmethod
    @tf.function
    def compute_gradients(images, model, class_index, deterministic=True):
        """
        Compute gradients for target class.
        Args:
            images : tf.tensor
                Gradient of model output at class index is computed with 
                respect to this tensor of images. Shape is (num_images, H, W, C)
            model : tf.keras.Model 
                tf.keras model to inspect
            class_index : int
                Index of targeted class
            deterministic : bool
                Whether to leave dropout on at prediction for calculation of
                gradients of class score with respect to input image for the 
                NN instantiated with weights from a single sample from 
                estimated Bayesian posterior (via dropout as bayesian 
                approximation)
        Returns:
            tf.Tensor: 4D-Tensor
            tf.tensor of gratients of logit at class_index with respect 
            to input images. Shape matches input (num_images, H, W, C)
        """
        num_classes = model.output.shape[1]

        expected_output = tf.one_hot([class_index] * images.shape[0], num_classes)

        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs, training=not deterministic)

            class_logits = predictions[:, class_index]
            
        # gradient of output at class_i with respect to inputs
        return tape.gradient(class_logits, inputs)

