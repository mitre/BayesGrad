import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_curve

"""

This library contains general functions related to model evaluation

"""


def plot_calibration_curve(probs, y_true, class_names, n_bins=10, title=None, save_path=None):
    """
    Plot calibration curve

    Parameters:
    -----------
    probs : np.array
        Predicted probabilities for each class. shape (n_predictions, n_classes)

    y_true : np.array
        True labels. shape (n_predictions,)

    class_names : list of strings
        List of class names. Class name at index i, should correspond with
        class i.

    n_bins : int
        Number of bins for calibration curve

    title : string
        Optional title for plot

    save_path : string
        Optional path for where to save plot. If None, plot not saved.
        
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    num_classes = len(set(y_true))
    # O v R
    for i in range(num_classes):
        y_prob_i = probs[:,i]
        y_true_i = np.empty_like(y_true)

        y_true_i[y_true == i] = 1
        y_true_i[y_true != i] = 0
        prob_true, prob_pred = calibration_curve(y_true_i, y_prob_i, normalize=False, n_bins=n_bins, strategy='uniform')
        ax.plot(prob_pred, prob_true, label=class_names[i])

    ax.plot(prob_true, prob_true, '--', color='grey', label='Perfect Calibration')
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("True Probability")
    if title:
        ax.set_title(title)
    plt.legend()
    if save_path:
        plt.savefig('/home/XNN4RAD/calibration_det.png', dpi=300)


def confusion_matrix(y_true, y_pred, scores=np.array([None]), pos_label=1, neg_label=0, title=None, filepath=None, print_scores=True):
    """
    Plot confusion matrix and print precision, recall, accuracy, specificity.
    For binary problem, if labels not 0 or 1 specify pos and negative labels.
    Currently only tested for binary case

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    scores : pd.Series
        Option to pass confidence metrics for prediction -- i.e. probabilities
        or non-thresholded decision values (as returned by decision_function
        on some sklearn classifiers).
    pos_label : int or string
        label to indicate positive class
    neg_label : int or string
        label to indicate negative class
    filepath : string
        Optional filepath to save image

    Returns
    -------
    None
    """
    confusion_matrix = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted']).T
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt='d')
    ax.set_ylim([0,2])
    if title:
        ax.set_title(title)
    if filepath:
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)


    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    recall = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    specificity = recall_score(y_true, y_pred, pos_label=neg_label)
    if scores.any():
        auc = roc_auc_score(y_true, scores)
    if print_scores:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"Specificity: {specificity}")
        if scores.any():
            print(f"AUC: {auc}")

