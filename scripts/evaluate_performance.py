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
import uncertainty_metrics.numpy as um
import numpy as np                                                            
import scipy.stats
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from codebase.data_utils_pet import load_dataset 
from codebase.bnn_utils import predict
from codebase.auc import AUC
from datetime import datetime
from functools import partial
import json
import copy


"""

    This script generates a results csv recording accuracy, ECE, and AUC for many
    retrainings of a BNN and its deterministic baseline according to the held out
    test set. Given that each BNN
    is trained using drop out for Bayesian approximation, the deterministic
    baseline is this same model with dropout turned off at prediction. We record
    model evaluation over many model retrainings in order to control for stochasticity
    in the model training process. The results file for the bayesian models is 
    saved to {FLAGS.output_path}/bayesian{FLAGS.suffix}.csv, while the results file 
    for the deterministic model is saved to {FLAGS.output_path}/bayesian{FLAGS.suffix}.csv. 
    trials_path is the path to a directory holding n trial directories, and 
    each trial directory contains a checkpoint directory with a saved model
    file. This file structure was created by running the job array
    run_train_VGGDrop_norm_abn_final_ja.sh. 
    A csv containing the predicted classification probabilities from each retraining
    of BNN and DNN is saved to {FLAGS.output_path}/bayesian_probs/probs{i}.csv
    and {FLAGS.output_path}/deterministic_probs/probs{i}.csv respectively.


"""

flags.DEFINE_string('trials_path', '',
                    'path to directory holding n trial directories each contianing a checkpoint directory containing a saved model file')
flags.DEFINE_string('output_path', '',
                    'Path to the directory where results csvs will be saved')
flags.DEFINE_string('suffix', '',
                    'Optional suffix to append to name of output files')
flags.DEFINE_string('data_path', '',
                    'Path to data directory')

FLAGS = flags.FLAGS


def main(argv):
    # load data
    # y_val norm/abn for evaluating predictions
    val = load_dataset(FLAGS.data_path, '17_segment', 'norm_abn', 'val')
    test = load_dataset(FLAGS.data_path, '17_segment', 'norm_abn', 'test')
    y_val = pd.concat((val['y'], test['y']))

    val = load_dataset(FLAGS.data_path, 'polar_plot', 'localization', 'val')
    test = load_dataset(FLAGS.data_path, 'polar_plot', 'localization', 'test')

    X_val = np.vstack((val['X'], test['X']))
    # just rest and stress
    X_val = X_val[:,:,:,:2] 


    # Dictionaries to store performance
    deterministic_perf = {'accuracy': [], 'ece': [], 'auc': []}
    bayesian_perf = {'accuracy': [], 'ece': [], 'auc': []}

    os.makedirs(f'{FLAGS.output_path}/bayesian_probs', exist_ok=True)
    os.makedirs(f'{FLAGS.output_path}/deterministic_probs', exist_ok=True)

    trials_path = FLAGS.trials_path
    for i, trial_dir in enumerate(os.listdir(trials_path)):
        trial_path = os.path.join(trials_path, trial_dir)
        for subdir in os.listdir(trial_path):
            if 'checkpoint' in subdir:
                checkpoint_path = os.path.join(trial_path, subdir)
        
        num_samples=150
        
        # Load model
        classifier = tf.keras.models.load_model(checkpoint_path)
        classifier.trainable = False
        classifier.compile(metrics=['accuracy', AUC()])
        
        # Deterministic
        probs_deterministic = predict(
        X_val.astype('float32'),
        classifier,
        logits=True,
        num_samples=None).reshape(-1)

        # Bayesian
        num_bins = 10
        probs_bayesian = predict(
            X_val.astype('float32'),
            classifier,
            logits=True,
            num_samples=num_samples,
            mean=True).reshape(-1)

        pd.Series(probs_deterministic).to_csv(f'{FLAGS.output_path}/deterministic_probs/probs{i}.csv') 
        pd.Series(probs_bayesian).to_csv(f'{FLAGS.output_path}/bayesian_probs/probs{i}.csv') 

        # Bayesian acc, ece, auc
        ece = um.ece(y_val, probs_bayesian, num_bins=num_bins)
        preds = np.empty_like(probs_bayesian)
        preds[probs_bayesian >= 0.5] = 1
        preds[probs_bayesian < 0.5 ] = 0
        acc = (preds == y_val).sum() / preds.shape[0]
        auc = roc_auc_score(y_val, probs_bayesian)
        
        bayesian_perf['accuracy'].append(acc)
        bayesian_perf['auc'].append(auc)
        bayesian_perf['ece'].append(ece)
        
        # Deterministic acc, ece, auc
        ece = um.ece(y_val, probs_deterministic, num_bins=num_bins)
        preds = np.empty_like(probs_deterministic)
        preds[probs_deterministic >= 0.5] = 1
        preds[probs_deterministic < 0.5 ] = 0
        acc = (preds == y_val).sum() / preds.shape[0]
        auc = roc_auc_score(y_val, probs_deterministic)
            
        deterministic_perf['accuracy'].append(acc)
        deterministic_perf['auc'].append(auc)
        deterministic_perf['ece'].append(ece)
        
    bayesian_perf = pd.DataFrame.from_dict(bayesian_perf)
    deterministic_perf = pd.DataFrame.from_dict(deterministic_perf)

    suffix = FLAGS.suffix
    if suffix != '':
        suffix = f"_{suffix}"

    bayesian_perf.to_csv(f'{FLAGS.output_path}/bayesian{suffix}.csv')
    deterministic_perf.to_csv(f'{FLAGS.output_path}/deterministic{suffix}.csv')

  


if __name__ == '__main__':
    app.run(main)
