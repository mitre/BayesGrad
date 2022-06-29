from __future__ import print_function
import sys
import os
BASE_DIR = os.getcwd().split('xnn4rad-pet')[0] + 'xnn4rad-pet/'
sys.path.append(BASE_DIR)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from tensorflow.keras.metrics import AUC, Precision, Recall
# from robustness_metrics.metrics.uncertainty import ExpectedCalibrationError as ECE
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import uncertainty_metrics as um
from sklearn.model_selection import train_test_split
# from codebase import mc_dropout_vgg_lld_2xf
from codebase import mc_dropout_vggish
from codebase.data_utils_pet import load_dataset
import codebase.custom_metrics as cm
from datetime import datetime
import argparse
from sklearn.metrics import roc_auc_score
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import io
# import tensorflow_probability

from functools import partial
from codebase.auc import AUC


"""
    Training loop adapted from https://github.com/geifmany/cifar-vgg/blob/master/cifar10vgg.py 
"""


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_pet(study, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(11,4.5))
        return_val = fig
    else:
        return_val = None
    
    sns.heatmap(study[:,:,0], ax=axs[0])
    axs[0].set_title('Stress')
    sns.heatmap(study[:,:,1], ax=axs[1])
    axs[1].set_title('Rest')
    
    
    return return_val


class PetVGG:
    def __init__(self,train=True, data_path='../data', lr=0.1, epochs=500, dropout_rate=0.15, reg=0.0, num_base_filters=64, std_scaler=0, vary_std_scaler=False, output_dir="./", problem='abnormal', channels=[True]*4, rotation_range=0, horizontal_flip=False, vertical_flip=False, width_shift_range=0, height_shift_range=0, suffix=None):
        self.lr = lr
        self.epochs = epochs
        self.dropout_rate=dropout_rate
        self.reg=reg
        self.num_base_filters = num_base_filters
        self.std_scaler=std_scaler
        self.num_classes = 2
        self.weight_decay = 0.0005
        self.channels = channels
        num_channels = channels.astype(int).sum()
        self.num_channels = num_channels
        self.x_shape = [48, 48, num_channels]
        self.output_dir=output_dir
        self.vary_std_scaler=vary_std_scaler
        self.problem = problem
        self.rotation_range=rotation_range
        self.horizontal_flip=horizontal_flip
        self.vertical_flip=vertical_flip
        self.width_shift_range=width_shift_range
        self.height_shift_range=height_shift_range
        self.suffix=suffix
        
        if problem == 'abn_localization':
            self.num_classes = 3
        else:
            self.num_classes = 2
        
        print('hflip: ', self.horizontal_flip,
              'vflip: ', self.vertical_flip)

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)


    @staticmethod
    def _augment(ex, std_scaler=0.01, vary_std_scaler=True):
        """
        Add random noise in data augmentation
        """
        # stress_scan, rest_scan, reserve, difference.
        ex = ex.copy()
        
        if vary_std_scaler:
            std_scaler = np.random.uniform(low=0.0, high=std_scaler) 

        for i in range(ex.shape[-1]):
            measurement = ex[:,:,i]
            std = std_scaler * np.abs(
                np.median(
                    measurement[measurement!=0].flatten()))
            mask = np.random.normal(0, std, size=measurement.shape)
            mask[measurement==0] = 0
            measurement += mask
            ex[:,:,i] = measurement
        return ex

        
    def build_model(self):
        """
        Return uncompiled VGGish model
        """
        model = mc_dropout_vggish.VGGDrop(input_shape=self.x_shape,
                                    num_classes=self.num_classes,
                                    num_base_filters=self.num_base_filters,
                                    learning_rate=None,
                                    dropout_rate=self.dropout_rate,
                                    l2_reg=self.reg)
        return model


    def train(self,model):
        """
        Run model training
        """

        # training parameters
        batch_size = 512
        maxepoches = self.epochs
#         maxepoches = 500
        learning_rate = self.lr
        lr_decay = 1e-6
        lr_drop = 20

        """
        Load data for different classification problems
        """

        # Order of polar plot channels:
        # stress, rest, reserve, difference
        if problem == 'abnormal':
            data = load_dataset('/q/PET-MBF/data',
                                'polar_plot',
                                'norm_abn',
                                'train',
                                val_col='nn_val_split')
            X, y = data['X'], data['y']
            print('channels:', self.channels)
            X = X[:,:,:,self.channels] 
            print('data_shape:', X.shape)
            
            X_train, y_train = X[data['val_split'] == 0], y[data['val_split'] == 0]
            X_val, y_val = X[data['val_split'] == 1], y[data['val_split'] == 1]
            class_weight = {0: 1.0, 1: 1.0}
            metrics =  ['accuracy', AUC()]

        else:
            data = load_dataset('/q/PET-MBF/data',
                                'polar_plot',
                                'localization',
                                'train',
                                val_col='nn_val_split')
            X, y = data['X'], data['y']
            print('channels:', self.channels)
            X = X[:,:,:,self.channels] 
            print('data_shape:', X.shape)

            if problem == 'scar':
                y = (y[['scar_lad', 'scar_rca', 'scar_lcx']].values.sum(axis=1) != 0).astype(int)
                class_weight = {0: 0.60902256, 1: 2.79310345}
                metrics =  ['accuracy', AUC()]
            elif problem == 'ischemia':
                y = (y[['ischemia_lad', 'ischemia_rca', 'ischemia_lcx']].values.sum(axis=1) != 0).astype(int)
                class_weight = {0: 1.0, 1: 1.46}
                metrics =  ['accuracy', AUC()]
            elif problem == 'abn_localization':
                y = y[['scar_lad', 'scar_rca', 'scar_lcx']].values | y[['ischemia_lad', 'ischemia_rca', 'ischemia_lcx']].values 
                metrics =  ['accuracy',
                            AUC(multi_label=True),
                            cm.make_metric_func_by_reg('lad',tf.keras.metrics.binary_accuracy),
                            cm.make_metric_func_by_reg('rca',tf.keras.metrics.binary_accuracy),
                            cm.make_metric_func_by_reg('lcx',tf.keras.metrics.binary_accuracy),
                            cm.Precision(from_logits=True, class_id=0, name='lad_precision'),
                            cm.Precision(from_logits=True, class_id=1, name='rca_precision'),
                            cm.Precision(from_logits=True, class_id=2, name='lcx_precision'),
                            cm.Recall(from_logits=True, class_id=0, name='lad_recall'),
                            cm.Recall(from_logits=True, class_id=1, name='rca_recall'),
                            cm.Recall(from_logits=True, class_id=2, name='lcx_recall')]
                class_weight = None

            X_train, y_train = X[data['val_split'] == 0], y[data['val_split'] == 0]
            X_val, y_val = X[data['val_split'] == 1], y[data['val_split'] == 1]
            



        # Build suffix string displaying the hyperparameter settings used to train the model
        # This will be used in where output files are stored
        channels = np.array([args.stress,
                             args.rest,
                             args.reserve,
                             args.difference])
        channel_abrvs = ['str', 'rst', 'rsv', 'dif']
        channels_str = ""
        for i, abrv in enumerate(channel_abrvs):
            if self.channels[i] == True:
                if i == 0:
                    channels_str += abrv
                else:
                    channels_str += f"_{abrv}"

        if not self.suffix:
            suffix = f'VGGDrop_{problem}_lr={self.lr}_drop={self.dropout_rate}_reg={self.reg}_base_filters={self.num_base_filters}_stdscaler={self.std_scaler}_vary={self.vary_std_scaler}_rotation={self.rotation_range}_hflip={self.horizontal_flip}_vflip={self.vertical_flip}_hshift={self.height_shift_range}_wshift={self.width_shift_range}_{channels_str}'
        else:
            suffix=self.suffix

        now = datetime.now().strftime("%Y%m%d-%H%M%S")
        tf_log_dir = os.path.join(self.output_dir,
                               "norm_abn_" + suffix,
                               "tensorboard" + now)
        file_writer = tf.summary.create_file_writer(os.path.join(tf_log_dir, 'plots'))

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)



        # data augmentation
        datagen = ImageDataGenerator(
            rotation_range=self.rotation_range,  # randomly rotate images (degrees, 0 to 180)
            horizontal_flip=self.horizontal_flip,  # randomly flip images
            vertical_flip=self.vertical_flip,  # randomly flip images
            width_shift_range=self.width_shift_range, # random horizontal shifts
            height_shift_range=self.height_shift_range, # random vertical shifts
            preprocessing_function=partial( # at gaussian noise to each example with std std_scalar
                self._augment,
                std_scaler=self.std_scaler,
                vary_std_scaler=self.vary_std_scaler)) # bool of whether std_scalar should vary example to example

        datagen.fit(X_train)


        # Save an example of training input to tensorboard
        batch = next(datagen.flow(
            data['X'], 
            data['y'], 
            batch_size=32, 
            shuffle=False))
        ex = batch[0][0]

        fig, ax = plt.subplots(3, 2, figsize=(11, 13.5))                           

        for i in range(3):                                                         
            batch = next(datagen.flow(                                             
                X_train,                                                         
                y_train,                                                         
                batch_size=32,                                                     
                shuffle=False))                                                    
            ex = batch[0][0]
            # Prepare the plot                                                     
            plot_pet(ex, ax[i]) 
            ax[i][0].set_ylabel(f'augmentation {i}')
        # Convert to image and log

        with file_writer.as_default():
          tf.summary.image("Training data", plot_to_image(fig), step=0)




        # Compile and fit model
        print('LEARNING RATE: ', learning_rate)
        opt = optimizers.Adam(learning_rate)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(
            loss=loss,
            optimizer=opt,
            metrics=metrics)


        checkpoint_filepath = os.path.join(self.output_dir,
                                           "norm_abn_" + suffix,
                                           "checkpoint" + now)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1)

    
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=tf_log_dir,
            update_freq="epoch",
            write_graph=False,
            histogram_freq=1)
        # training process in a for loop with learning rate drop every 25 epoches.

        # for normalization
        callbacks=[model_checkpoint_callback,
                  tensorboard_callback]
        historytemp = model.fit(datagen.flow(X_train, y_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             seed=0),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=datagen.flow(X_val, y_val),
                            callbacks=callbacks,
                            verbose=2,
                            class_weight=class_weight)
        return model

def parse_args():
    """
    Parse command line arguments
    """

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str,
                        help="""Directory where data is stored""")
    parser.add_argument('--output_dir', type=str,
                        help="""Base directory for tensorboard logs and
                             checkpoints""")
    parser.add_argument('--lr', type=float,
                        help="""Learning Rate""")
    parser.add_argument('--epochs', type=int,
                        help="""Epochs""")
    parser.add_argument('--dropout_rate', type=float,
                        help="""Drop prob""")
    parser.add_argument('--reg', type=float,
                        help="""reg""")
    parser.add_argument('--num_base_filters', type=float,
                        help="""reg""")
    parser.add_argument('--preproc_std_scaler', type=float,
                        help="""Scaler to multiply median of measurement to get
                             STD of noise added to channel as augmentation""")
    parser.add_argument('--preproc_rotation_range', type=float,
                        help="""Data augmentation rotation, 0-180 degrees""")
    parser.add_argument('--preproc_horizontal_flip', action='store_true', default=False,
                        help="""Data augmentation horizontal flip""")
    parser.add_argument('--preproc_vertical_flip', action='store_true', default=False,
                        help="""Data augmentation vertical flip""")
    parser.add_argument('--preproc_width_shift_range', type=float,
                        help="""Data augmentation width shift, 0-1""")
    parser.add_argument('--preproc_height_shift_range', type=float,
                        help="""Data augmentation height shift, 0-1""")

    parser.add_argument('--vary_std_scaler', type=bool,
                        help="""If true, scaler for std for preprocessing will
                                be drawn from uniform dist in range 0.0 to
                                preproc_std_scaler for each example""")
    parser.add_argument('--problem',
                        type=str,
                        choices=['abnormal', 'scar', 'ischemia', 'abn_localization'],
                        help="""'abnormal', 'scar' or 'ischemia'""")
    parser.add_argument('--suffix',
                        type=str,
                        help="""Optional suffix for output""")
    parser.add_argument('--biased', action='store_true',
                        help="""Train on biased data including only scans from
                             male patients""")
    parser.add_argument('--rest', action='store_true',
                        help="""Include rest channel in input data""")
    parser.add_argument('--stress', action='store_true',
                        help="""include stress channel in input data""")
    parser.add_argument('--reserve', action='store_true',
                        help="""include reserve channel in input data""")
    parser.add_argument('--difference', action='store_true',
                        help="""include difference channel in input data""")



    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('args.lr: ', args.lr)
    print('preproc_horizontal_flip: ', args.preproc_horizontal_flip)
    print('args.preproc_vertical_flip: ', args.preproc_vertical_flip)

    lr = args.lr
    epochs = args.epochs
    dropout_rate = args.dropout_rate
    reg = args.reg
    num_base_filters = args.num_base_filters
    std_scaler = args.preproc_std_scaler
    vary_std_scaler = args.vary_std_scaler
    vary_std_scaler=False
    rotation_range=args.preproc_rotation_range
    horizontal_flip=args.preproc_horizontal_flip
    vertical_flip=args.preproc_vertical_flip
    height_shift_range=args.preproc_height_shift_range
    width_shift_range=args.preproc_width_shift_range
    suffix=args.suffix

    output_dir = args.output_dir
    data_path = args.data_path
    problem = args.problem
    channels = np.array([args.stress,
                         args.rest,
                         args.reserve,
                         args.difference])
    print('vary_std_scaler:', vary_std_scaler)

    # Build and fit model
    model = PetVGG(
        train=True,
        data_path=data_path,
        lr=lr,
        epochs=epochs,
        dropout_rate=dropout_rate,
        reg=reg,
        num_base_filters=num_base_filters,
        std_scaler=std_scaler,
        vary_std_scaler=vary_std_scaler,
        output_dir=output_dir,
        problem=problem,
        channels=channels,
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        height_shift_range=height_shift_range,
        width_shift_range=width_shift_range,
        suffix=suffix)

