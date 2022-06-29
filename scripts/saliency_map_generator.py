import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad-pet')[0] + 'xnn4rad-pet/'
sys.path.append(BASE_DIR)

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
from tensorflow.python.keras.datasets.cifar import load_batch
from codebase.data_utils_pet import load_dataset 
from codebase.vanilla_gradients import VanillaGradients
from codebase.bayesgrad import BayesGrad
from codebase.bnn_utils import predict
from datetime import datetime
from functools import partial
from codebase.smoothgrad import SmoothGrad
import json
import copy

"""

    This script takes a trained model and generates and saves attention maps
    from the given model on the given train, validation, or test split. 

    If it does not already exist, a directory will be created at:
        {FLAGS.output_dir}/{FLAGS.model_alias}/{FLAGS.split}/

    The following files will be saved in this driectory:
    
        {FLAGS.smethod}.json
        A json file containing containing attention maps corressponding to 
        the study numbers in study_nos.csv. This saves data from a 4d array
        with dimension (number of examples, H, W, number of channels) where
        number of channels is 2 for rest and stress measurements. 

        'study_nos.csv'
        A parallel csv containing study numbers cooresponding to the attention
        maps in {FLAGS.smethod}.json

        'probabilities.csv'
        A parallel csv containing predicted probabilities for classification
        corresponding to each attention map in {FLAGS.smethod}.csv. This output
        file is generated and saved only if the 'probabilities' flag is true.

"""

flags.DEFINE_string('data_path', '',
                    'Path to directory containing data')
flags.DEFINE_string('model_path', '',
                    'Path to model used to derive saliency maps')
flags.DEFINE_string('model_alias', '',
                    "Intermediate dir_name to hold model's smaps. This should be a name for the model being used")
flags.DEFINE_string('output_dir', '',
                    'The base directory where the saliency maps are stored.')
flags.DEFINE_string('split', '',
    """Valid options include 'train', 'validation_hp', 'validation_mc',
    'test'""")
flags.DEFINE_string('smethod', '',
                    """Method to generate saliency maps. Valid options include 'BG', 'BG_2', 'BG_STD', 'BG_VAR', 'SH', 'SH_2', 'SG', 'SG_2', 'SH_VAR'""")
flags.DEFINE_bool('probabilities', 'False',
                    'Whether to output predicted probabilities.')

FLAGS = flags.FLAGS


def load_data():
    """
    """     
    if (FLAGS.split == 'train') or (FLAGS.split == 'validation_hp'):
        data = load_dataset(FLAGS.data_path, 'polar_plot', 'norm_abn', 'train', val_col='nn_val_split')
        X, y = data['X'], data['y']          

        # just rest and stress
        x = x[:,:,:,:2]
        
        if FLAGS.split == 'train':
            split = 0
        else:
            split = 1

        split_mask = data['val_split'] == split 
        X, y = X[split_mask], y[split_mask]  
        study_nos = data['study_no'].values[split_mask]
    elif FLAGS.split == 'validation_mc':
        data = load_dataset(FLAGS.data_path, 'polar_plot', 'norm_abn', 'val')
        X, y, study_nos = data['X'], data['y'], data['study_no']
        X = X[:,:,:,:2]
    elif FLAGS.split == 'test':
        data = load_dataset(FLAGS.data_path, 'polar_plot', 'norm_abn', 'test')
        X, y, study_nos = data['X'], data['y'], data['study_no']
        X = X[:,:,:,:2]
    else:
        raise ValueError("split must be 'train', 'validation_hp', 'validation_mc', or 'test'")

    return {'X': X, 'y': y, 'study_no': study_nos}

def write_output(smaps, study_nos, preds=None):
    smaps_copy = []
    for smap in smaps:
        smaps_copy.append(smap.tolist())
        
    basepath = os.path.join(
        FLAGS.output_dir,
        FLAGS.model_alias,
        FLAGS.split)
    os.makedirs(basepath, exist_ok=True)    

    filepath = os.path.join(
        basepath,
        f"{FLAGS.smethod}.json") 

    print("SAVING TO: ", filepath)
    with open(filepath, "w") as outfile:
        json.dump(smaps_copy, outfile)

    study_no_path = os.path.join(basepath,
        'study_nos.csv')
    study_nos = pd.Series(study_nos)
    study_nos.to_csv(study_no_path, index=False)

    if preds is not None:
        probs_path = os.path.join(basepath,
            'probabilities.csv')
        pd.Series(preds).to_csv(probs_path, index=False)



def main(argv):
    # Load model
    keras.backend.clear_session()
    classifier = tf.keras.models.load_model(FLAGS.model_path)

    classifier.trainable = False
    classifier.compile(metrics=['accuracy'])
    
    data = load_data()
    X, study_nos = data['X'], data['study_no']
    num_smaps = len(study_nos)

    num_samples = 100
    smaps = []

    # Generate appropriate attention map for each example
    for i in range(num_smaps):
        if i % 10 == 0:
            logging.info(f"{i} of {num_smaps}")

        image = X[i]
        study_no = study_nos[i]
        data = np.array([image])
        data = tf.convert_to_tensor(data)
        
         
        if FLAGS.smethod == 'BG':
            explainer = BayesGrad()
            bg_smap = explainer.explain(data,
                                        classifier,
                                        class_index=0,
                                        num_samples=num_samples,
                                        uncertainty=None,
                                        squared=False)[0]
            smaps.append(bg_smap)
        elif FLAGS.smethod == 'BG_2': 
            explainer = BayesGrad()
            bg_2_smap = explainer.explain(data,
                                          classifier,
                                          class_index=0,
                                          num_samples=num_samples,
                                          uncertainty=None,
                                          squared=True)[0]
            smaps.append(bg_2_smap)
        elif FLAGS.smethod == 'BG_STD': 
            explainer = BayesGrad()
            bg_std_smap = explainer.explain(data,
                                            classifier,
                                            class_index=0,
                                            num_samples=num_samples,
                                            uncertainty='std',
                                            squared=False)[0]
            smaps.append(bg_std_smap)
        elif FLAGS.smethod == 'BG_VAR': 
            explainer = BayesGrad()
            bg_var_smap = explainer.explain(data,
                                            classifier,
                                            class_index=0,
                                            num_samples=num_samples,
                                            uncertainty='var',
                                            squared=False)[0]
            smaps.append(bg_var_smap)
        elif FLAGS.smethod == 'SH': 
            explainer = VanillaGradients()
            vg_smap = explainer.explain(data,
                                        classifier,
                                        class_index=0,
                                        squared=False)[0]
            smaps.append(vg_smap)
        elif FLAGS.smethod == 'SH_2':
            explainer = VanillaGradients()
            vg_2_smap = explainer.explain(data,
                                        classifier,
                                        class_index=0,
                                        squared=True)[0]
            smaps.append(vg_2_smap)
        elif FLAGS.smethod == 'SG': 
            explainer = SmoothGrad()
            sg_smap = explainer.explain(
                data,
                classifier,
                class_index=0,
                num_samples=num_samples,
                stdev_spread=.05,
                uncertainty=None,
                squared=False,
                debug=False)[0]
            smaps.append(sg_smap)
        elif FLAGS.smethod == 'SG_2':
            explainer = SmoothGrad()
            sg2_smap = explainer.explain(
                data,
                classifier,
                class_index=0,
                num_samples=num_samples,
                stdev_spread=.05,
                uncertainty=None,
                squared=True,
                debug=False)[0]
            smaps.append(sg2_smap)
        elif FLAGS.smethod == 'SH_VAR': 
            explainer = SmoothGrad()
            sh_var_smap = explainer.explain(
                data,
                classifier,
                class_index=0,
                num_samples=num_samples,
                stdev_spread=.05,
                uncertainty='var',
                squared=False,
                debug=False)[0]
            smaps.append(sh_var_smap)
        else:
            raise NotImplementedError(f"Invalid value {FLAGS.smethod} for smethod")

    # Optionally get predicted probabilities for each example
    if FLAGS.probabilities:
        preds = predict(X,
            classifier,
            num_samples=num_samples,
            mean=True,
            logits=True)
        preds = preds.reshape(-1)
    else:
        preds=None

    write_output(smaps, study_nos, preds)


if __name__ == '__main__':
    app.run(main)
