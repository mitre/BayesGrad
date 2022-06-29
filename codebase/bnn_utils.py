import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad-pet')[0] + 'xnn4rad-pet/'
sys.path.append(BASE_DIR)

import numpy as np
import scipy
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

"""

This library contains functions to facilitate prediction with MC dropout.
The predict function produces an array of predictions from samples of the 
bayesian posterior (i.e. multiple forward passes when dropout is left on at
prediction). In order to take advantage of efficiencies in the existing model
code, the function modifies the internal predict step of the keras model
to leave dropout on before making the predictions. It changes the internal
predict step back to its original process before the function returns so that
the keras model is not modified outside of this function.

"""

def _toggle_predict_step(model, dropout):  
  from tensorflow.keras import Sequential

  model.trainable = False
  model.compile(metrics=['accuracy'])

  predict_step = _make_predict_step(dropout=dropout)
                                                                                 
  model.predict_step = predict_step.__get__(model, Sequential)                   
  return model


def _make_predict_step(dropout):
  from tensorflow.python.keras.engine import data_adapter

  def predict_step(self, data):                                                  
     """The logic for one inference step. This function is overridden to        
     leave training on for Bayesian prediction through dropout.                 


     This method can be overridden to support custom inference logic.           
     This method is called by `Model.make_predict_function`.                    
     This method should contain the mathematical logic for one step of inference.
     This typically includes the forward pass.                                  
     Configuration details for *how* this logic is run (e.g. `tf.function` and
     `tf.distribute.Strategy` settings), should be left to                      
     `Model.make_predict_function`, which can also be overridden.               

     Args:                                                                      
       data: A nested structure of `Tensor`s.                                   
     Returns:                                                                   
       The result of one inference step, typically the output of calling the 
       `Model` on data.                                                         
     """                                                                        
     data = data_adapter.expand_1d(data)                                        
     x, _, _ = data_adapter.unpack_x_y_sample_weight(data)   
     return self(x, training=dropout)
  
  return predict_step

def predict(x, model, num_samples, mean=False, logits=True):                    
  """                                                                           
  x :                                                                           
  Data to use for prediction                                                  

  model : keras or tf model                                                     
  Model to make predictions                                                   

  num_samples : int                                                             
  Number of samples from the bayesian posterior. If None, then return         
  deterministic prediction.                                                   

  mean : bool                                                                   
  If True return the mean of predicted probabilities from each forward pass.  
  If False return a matrix with probabilities from all num_samples forward passes

  logits : bool                                                                 
  Whether the model has logits as output layer. If False, output layer should 
  be softmax(logits) or sigmoid(logits). If True, predict will apply softmax 
  or sigmoid depending on the output dimension of the network.                                                        


  Returns: 
  --------

  When num_samples is not None and mean is False:
      Return an array of shape (num_samples, num_examples, nn_output_dimension).
      Each entry in num_samples is predicted probabilities for each input example
      for a single forward pass with dropout left on.

  When num_samples is not None and mean is True:
      Return an array of shape (num_examples, nn_output_dimension).
      Output is the mean of probabilities across predictions from all samples
      from Bayesian posterior. 

  When num_samples is None:
      Return an array of shape (num_examples, nn_output_dimension).
      Output is the predicted probabilities when dropout is left off at
      (i.e. this is a deterministic version of the model).

  """                                                                           
                                                                              
  if model.layers[-1].output_shape[1] == 1:
    activation_b = scipy.special.expit
    activation_d = scipy.special.expit
  else:
    # Shape of predictions (num bayesian samples, num_examples, num_classes)
    activation_b = partial(scipy.special.softmax, axis=2)
    # Shape of predictions (num_examples, num_classes)
    activation_d = partial(scipy.special.softmax, axis=1)

                                                                              
  if num_samples:
    # Bayesian
    model = _toggle_predict_step(model, dropout=True)
    preds = np.asarray([model.predict(x) for _ in range(num_samples)])
    # Toggle back to deterministic so don't change model behavior
    # outside of function
    model = _toggle_predict_step(model, dropout=False)
    
    if logits:
      preds = activation_b(preds)

    if mean:                                                                    
      preds = preds.mean(axis=0)                                                
                                                                            
  else:
    # Deterministic
    model = _toggle_predict_step(model, dropout=False)
    preds =  model.predict(x)
    
    if logits:
      preds = activation_d(preds)
    else:                                                                       
      preds = preds
          
  return preds

