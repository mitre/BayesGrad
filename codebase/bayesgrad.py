import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad-pet')[0] + 'xnn4rad-pet/'
sys.path.append(BASE_DIR)

import tensorflow as tf
import numpy as np
from codebase.vanilla_gradients import VanillaGradients

"""

Implementation of BayesGrad and BayesGrad derivatives (i.e. BayesGrad^2,
BayesGrad). This package expects that the model's 
last layer is logits with no activation function applied. This package
expects that the model is intended to be used as a BNN trained with
dropout as Bayesian approximation.


Adapted from tf-explain
https://github.com/sicara/tf-explain

"""

class BayesGrad:
    def __init__(self):
        self.base_explainer = VanillaGradients()

    def explain(self, data, model, class_index, num_samples, uncertainty=None, squared=False):
        """
        Compute BayesGrad for a specific class index

        Args:
            images : tf.tensor
                tensor of images to perform the explainer method on
                Shape is (num_images, H, W, C)
            model : tf.keras.Model
                tf.keras model to inspect
            class_index : int
                Index of targeted class
            num_samples : int
                Output is calculated by aggregating num_samples vanilla gradient
                calculations based on num_samples instantiations of the model
                from weights drawn from the Bayesian posterior (i.e. dropout
                left on at inference)
            uncertainty : str
                Optional argument to take variance or standard deviation of
                gradient feature maps from many BNN samples rather than average.
                Valid options are 'std', 'var', and None. If uncertainty is 
                None, than function returns the mean of the noisy gradients.
            squared : bool
                Whether to square gradients
                
        Returns:
            np.array of BayesGrad gratients. Shape matches input
            shape (num_images, H, W, C).
        """
        s_maps = []
        
        for _ in range(num_samples):
            s_map = self.base_explainer.explain(data, model, class_index, deterministic=False, squared=squared) 
            s_maps.append(s_map)

        s_maps = np.asarray(s_maps)
        


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
