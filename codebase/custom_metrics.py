import tensorflow as tf
from tensorflow import keras


class Precision(tf.keras.metrics.Precision):
    """
    Version of precision Keras metric that accommodate from_logits param for 
    models which do not end in activation layer
    """
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)


class Recall(tf.keras.metrics.Recall):
    """
    Version of recall Keras metric that accommodate from_logits param for 
    models which do not end in activation layer
    """
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Recall, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)



def make_metric_func_by_reg(region, keras_metric):
    """
    Metrics related to localization baseline.

    make_metric_func_by_reg takes a region and a keras metric and returns a 
    keras metric which gives a score according to prediction of the specified
    region
    """

    if region == 'lad':
        i = 0
    elif region == 'rca':
        i = 1
    elif region == 'lcx':
        i = 2

    def metric_by_reg(y_true, y_pred):
        true = y_true[:,i]
        pred = y_pred[:,i]
        return keras_metric(true, pred)
    metric_by_reg.__name__ = f"{region}_" + keras_metric.__name__
    return metric_by_reg
