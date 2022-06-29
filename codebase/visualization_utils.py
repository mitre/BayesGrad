from matplotlib import pylab as P
import matplotlib.pyplot as plt
# from pretrained_keras_models.vgg16 import VGG16

import numpy as np
from scipy.ndimage.filters import gaussian_filter

"""

Functions related to visualizing saliency maps

"""



"""
These functions come from the python saliency package:
https://github.com/pair-code/saliency


# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
def VisualizeImageGrayscale(image_3d, percentile=99):
    r"""Returns a 3D tensor as a grayscale 2D tensor.
    This method sums a 3D tensor across the absolute value of axis=2, and then
    clips values at a given percentile.
    """
    image_2d = np.sum(np.abs(image_3d), axis=2)

    vmax = np.percentile(image_2d, percentile)
    vmin = np.min(image_2d)

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

def VisualizeImageDiverging(image_3d, percentile=99):
    r"""Returns a 3D tensor as a 2D tensor with positive and negative values.
    """
    image_2d = np.sum(image_3d, axis=2)

    span = abs(np.percentile(image_2d, percentile))
    vmin = -span
    vmax = span

    return np.clip((image_2d - vmin) / (vmax - vmin), 0, 1)

"""
These functions come from code released with the paper
"Sanity Checks for Saliency Maps"
https://github.com/adebayoj/sanity_checks_saliency

Some functions have been edited
"""
def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        print('==2')
        img = np.absolute(img)
        img = img/float(img.max())
    else:
        img = VisualizeImageGrayscale(img)
    return img


def diverging_norm(img):
    """Returns image with positive and negative values."""

    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        imgmax = np.absolute(img).max()
        img = img/float(imgmax)
    else:
        img = VisualizeImageDiverging(img)
    return img

def plot_single_img(img,
                    ax=False,
                    norm=diverging_norm,
                    show_axis=False,
                    grayscale=False,
                    cmap='gray',
                    title='',
                    fig_size=(4, 4)):
    """Function to plot a single image."""

    plt.figure(figsize=fig_size)
    if norm:
        img = norm(img)
    if not show_axis:
        plt.axis('off')
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.show()
    

def abs_grayscale_norm(img):
    """Returns absolute value normalized image 2D."""
    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        img = np.absolute(img)
        img = img/float(img.max())
    else:
        img = VisualizeImageGrayscale(img)
    return img

def diverging_norm(img):
    """Returns image with positive and negative values."""

    assert isinstance(img, np.ndarray), "img should be a numpy array"
    shp = img.shape
    if len(shp) < 2:
        raise ValueError("Array should have 2 or 3 dims!")
    if len(shp) == 2:
        imgmax = np.absolute(img).max()
        img = img/float(imgmax)
    else:
        img = VisualizeImageDiverging(img)
    return img

def normalize_image(x):
    x = np.array(x).astype(np.float32)
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)
    return x_norm

def visualize_smap(smap, ax=None, title=None, style='gs', prob=None):
    """
    Parameters:
    -----------
    smap : np.array
        The saliency output
    
    axis : matplotlib axis
        Optional axis to plot figure on

    title : string
        Optional title for plot

    style : string
        Type of smap visualization. Valid options include 'gs' for grayscale
        'color' for color visualization of absolute value of saliency map
        or 'div' for diverging -- preserves positive and negative gradients
    """
    weight=prob
    if not ax:
        fig, ax = plt.subplots()

    if title:
        ax.set_title(title)

    if style == 'gs':
        smap = abs_grayscale_norm(smap)
        if not prob:
            ax.imshow(smap,'gray')
        else:
            ax.imshow(smap * weight,'gray')
    elif style == 'color':
        smap = abs_grayscale_norm(smap)
        if not prob:
            plot = ax.imshow(smap,'viridis')
        else:
            plot = ax.imshow(smap*weight,vmin=0, vmax=1, cmap='viridis')
    elif style == 'div':
        smap = diverging_norm(smap)
        if not prob:
            ax.imshow(smap, vmin=0, vmax=1, cmap='bwr')
        else:
            ax.imshow(smap * weight, vmin=0, vmax=1, cmap='bwr')
    
