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
from matplotlib import patches
from matplotlib import colors
import seaborn as sns
from collections import defaultdict
from copy import deepcopy
import json

import codebase.visualization_utils as vu
import codebase.data_utils_pet as dup

"""

This file contains functions related to the evaluation and visulaization
 of attention maps for the PET abnormality detection problem. 

"""



def pixels_to_vessel(smap, top_percent, vessel_mask, importance='abs', plot=False, title=None, ax=None, legend=True):
    """
    This function takes a smap, top_percent, a vessel mask, and return the fraction of the 
    top_percent of total pixels that fall within each vessel territory.
    The top X percent of the most important pixels from the saliency map are 
    considered. Each pixel is mapped to the vessel territory in which
    the pixel falls according to the vessel mask. This function returns the
    fraction of pixels in top_percent which belong to each vessel territory.
    This is used in the calculation of top K predict which is described in the
    manuscript. 

    smap : np.array
        Saliency map of polar map representation of PET-MBF image.

    top_percent : int
        Top X percent of most important pixels from the saliency map to be 
        considered. Each pixel will be mapped to the vessel territory in which
        the pixel falls according to the vessel mask. 
        
    vessel_mask : np.array
        Vessel mask used to assign pixels to vessel territories. 1=RCA, 2=LAD,
        3=LCX.

    importance : str
        importance determines how pixel importance rankings are calculated. 
        Possible values include 'pos', 'neg', or 'abs'. 
        If 'pos' is chosen, the most important pixels in the smap are those
        with the largest signed value. If 'neg' is chosen, the most important
        pixels in the smap are those with the smallest signed value. If 
        'abs' is chosen, the most important pixels in the smap are those with
        the largest absolute value. It is recommended to use 'abs'
    
    plot : bool
        Whether to plot the masked top_percent of pixels

    title : string
        Optional title of plot

    ax : matplotlib axis
        Optional axis for plot

    legend : bool
        Whether to show legend of vessel territories
    """
    top_pix = int(top_percent/100 * 48*48)
    smap_mean = smap.mean(axis=2)
    
    if importance == 'abs':
        smap_mean = np.abs(smap_mean)
        ranked_pixels = np.flip(smap_mean.reshape(-1).argsort())
    elif importance == 'pos':
        ranked_pixels = np.flip(smap_mean.reshape(-1).argsort())
    elif importance == 'neg':
        ranked_pixels = smap_mean.reshape(-1).argsort()
        
    top = ranked_pixels[:top_pix]
    bottom = ranked_pixels[top_pix:]

    smap_flatten = smap_mean.copy().reshape(-1)
    smap_flatten[top] = 1
    smap_flatten[bottom] = 0
    
    smap_mask = smap_flatten.reshape(48, 48)

    mapped_mask = (smap_mask * vessel_mask)
    if plot:
        plot_mapped_mask(mapped_mask, title=title, ax=ax, legend=legend)
    
    counts = {}
    counts['rca'] = (mapped_mask.flatten() == 1).sum()
    counts['lad'] = (mapped_mask.flatten() == 2).sum()
    counts['lcx'] = (mapped_mask.flatten() == 3).sum()
    counts = pd.Series(counts)
    counts = counts / counts.sum()
    return counts


from matplotlib.patches import Patch
def plot_region(mapped_mask, lad=False, rca=False, lcx=False, title=None, ax=None):
    """
    This function takes the vessel mask output by make_20r_vessel_mask 
    and boolean flags for each of the three vessel territories, and it plots
    a mask with all territories with flags set to True in yellow, and all 
    territories with flags set to False in grey
    """
    
    region_color = np.array([256,244,4]) / 255
    region_color_mask = (region_color.reshape(1,1,3) * np.ones(shape=(48,48,3)))

    else_color = np.array([166,166,166]) / 255 
    else_color_mask = (else_color.reshape(1,1,3) * np.ones(shape=(48,48,3)))
    
    color_mask = np.zeros(shape=(48, 48, 3))

    color_mask[mapped_mask == 1] = else_color
    color_mask[mapped_mask == 2] = else_color
    color_mask[mapped_mask == 3] = else_color

    if lad:
        color_mask[mapped_mask == 2] = region_color
    if rca:
        color_mask[mapped_mask == 1] = region_color
    if lcx: 
        color_mask[mapped_mask == 3] = region_color
    
    if not ax:
        fig, ax = plt.subplots()
    
    ax.imshow(color_mask)
    ax.set_axis_off()

    if title:
        ax.set_title(title)


def plot_mapped_mask(mapped_mask, title=None, ax=None, legend=True):
    """
    This function takes the vessel mask output by make_20r_vessel_mask 
    and plots the mask using MITRE colors for the 3 vessels
    """
    lad_color = np.array([162,202,237]) / 255
    lad_color_mask = (lad_color.reshape(1,1,3) * np.ones(shape=(48,48,3)))

    rca_color = np.array([0,91,147]) / 255
    rca_color_mask = (rca_color.reshape(1,1,3) * np.ones(shape=(48,48,3)))

    lcx_color = np.array([166,166,166]) / 255 
    lcx_color_mask = (lcx_color.reshape(1,1,3) * np.ones(shape=(48,48,3)))
    
    color_mask = np.zeros(shape=(48, 48, 3))

    color_mask[mapped_mask == 1] = rca_color
    color_mask[mapped_mask == 2] = lad_color
    color_mask[mapped_mask == 3] = lcx_color
    

    if not ax:
        fig, ax = plt.subplots()
    
    ax.imshow(color_mask)
    ax.set_axis_off()

    if legend:
        legend_elements = [Patch(facecolor=lad_color,
                             label='LAD'),
                       Patch(facecolor=lcx_color,
                             label='LCX'),
                       Patch(facecolor=rca_color,
                             label='RCA')]
#         ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1))
        ax.legend(handles=legend_elements )

    
    if title:
        ax.set_title(title)


def topK_predict(smaps, k, y, vessel_mask, vessel_to_int, importance='abs', plot=False, study_nos=None):
    """
    smap : np.array
        Array of saliency maps
        
    k : int
        The top percent of most important pixels to consider
        
    y : pd.DataFrame
        The labels
        
    vessel_mask : pd.DataFrame
    
    absolute : bool
        Whether to take the absolute value of the vessel map
    """
    int_to_vessel = dict(zip([0,1,2], ['lad', 'rca', 'lcx']))    


    vessel_mask = vessel_mask.copy()
    
    results = []
    for i in range(smaps.shape[0]):
        y_i = y[i]
        smap_i = smaps[i]
        
        if plot:
            fig, ax = plt.subplots()
            vessel_assignments = pixels_to_vessel(smap_i,
                k,
                vessel_mask,
                importance=importance,
                plot=True,
                ax=ax)
            pred = vessel_assignments.index[vessel_assignments.argmax()]
            perc = vessel_assignments[vessel_assignments.argmax()] 
            pred_enc = vessel_to_int[pred]
            if study_nos is not None:
                ax.set_title(f"{study_nos[i]} {pred}")
            else:
                ax.set_title(f"pred: {pred} perc: {'{:.3f}'.format(perc)} True: {int_to_vessel[y_i]}")
            plt.plot() 
                
        else:
            vessel_assignments = pixels_to_vessel(smap_i,
                k,
                vessel_mask,
                importance=importance)
            pred = vessel_assignments.index[vessel_assignments.argmax()]
            pred_enc = vessel_to_int[pred]
        results.append(pred_enc)
        
    return np.array(results)


def make_20r_vessel_mask(lad_path, rca_path, lcx_path):
    """
    lad_path : str
        Path to CSV containing mask for 20 ring polar plot where areas
        corresponding to the LAD are denoted with 1 and all other areas 
        are denoted with 0
    rca_path : str
        Path to CSV containing mask for 20 ring polar plot where areas
        corresponding to the RCA are denoted with 1 and all other areas 
        are denoted with 0

    lcx_path : str
        Path to CSV containing mask for 20 ring polar plot where areas
        corresponding to the LCX are denoted with 1 and all other areas 
        are denoted with 0

    Return a mask of 20 ring polar plot where RCA is denoted with 1, LAD is 
    denoted with 2, and LCX is denoted with 3. This function reads in individual
    masks for each territory generated with flowquant and saved as CSVs, and it
    fills in ambiguous pixels where flowquant mask interpolated resulting in
    non-integer mask values with the value corresponding to the closest region.
    """
    mask_lad = pd.read_csv(lad_path, header=None).values
    mask_rca = pd.read_csv(rca_path, header=None).values
    mask_lcx = pd.read_csv(lcx_path, header=None).values

    mask_lad[mask_lad == 1] = 2
    mask_lcx[mask_lcx == 1] = 3

    mask_lad[(mask_lad != 2) & (mask_lad != 0)] = -1* mask_lad[(mask_lad != 2) & (mask_lad != 0)]
    mask_rca[(mask_rca != 1) & (mask_rca != 0)] = -1* mask_rca[(mask_rca != 1) & (mask_rca != 0)]
    mask_lcx[(mask_lcx != 3) & (mask_lcx != 0)] = -1* mask_lcx[(mask_lcx != 3) & (mask_lcx != 0)]
    mask = mask_lad + mask_lcx + mask_rca



    # LAD
    mask[22:24,4:16] = 2
    mask[22:24,4:16] = 2
    mask[21,5:16] = 2
    mask[27,18:22] = 2
    mask[28,15:20] = 2
    mask[29,16:18] = 2
    mask[27,27] = 2
    mask[:,34:][mask[:,34:] < 0] = 0

    # LCX
    mask[7,34] = 3
    mask[8,33:36] = 3
    mask[9:11,32:36] = 3
    mask[11:13,31:36] = 3
    mask[13,30:36] = 3
    mask[14:16,29:36] = 3
    mask[16,30:36] = 3
    mask[17:19,31:36] = 3
    mask[18,30] = 3
    mask[19:27,29:36] = 3
    mask[20:23,28:36] = 3
    mask[26:28,28:36] = 3
    mask[32,28:36] = 3
    mask[:24,:34][mask[:24,:34] < 0] = 0

    # RCA
    mask[29:36,22:29] = 1
    mask[34:40,29] = 1
    mask[35:43,30] = 1
    mask[37:42,31] = 1
    mask[39:41,32] = 1
    mask[40,33] = 1
    mask[24:34,13] = 1
    mask[26:34,14] = 1
    mask[29:34,15] = 1
    mask[28,20:22] = 1
    mask[28,26:28] = 1
    mask[29:32,29] = 1
    mask[24:,:34][mask[24:,:34] < 0] = 0

    return(mask)


def plot_vessel_boundries(ax, rings, color='black'):
    """
    Takes a matplotlib axis object and number of rings in scan and overlays 
    vessel boundries on top of what is already on the axis object. This is
    used to superimpose the outline of the vessel segmentation on visualizations
    of attention maps to help show which regions are indicated as most important
    according to the attention map.
    """
    
    r_24ring = 46
    r_16ring = r_24ring * (rings/24)
    segment_r = r_16ring / 4.0
    center = (48/2, 48/2)
    
    apex_b1 = patches.Arc(center,
                          segment_r,
                          segment_r,
                          linewidth=2,
                          angle=-45,
                          theta1=0,
                          theta2=180,
                          color=color)

    LAD_mid1 = patches.Arc(center,
                          segment_r*2,
                          segment_r*2,
                          linewidth=2,
                          angle=360 *3/8,
                          theta1=0,
                          theta2=360/8,
                          color=color)
    LAD_mid2 = patches.Arc(center,
                          segment_r*2,
                          segment_r*2,
                          linewidth=2,
                          angle=-1/6*360,
                          theta1=0,
                          theta2=360*3/8 - 1/3*360,
                          color=color)
    LCX_mid1 = patches.Arc(center,
                          segment_r*2,
                          segment_r*2,
                          linewidth=2,
                          angle=0,
                          theta1=1/8*360,
                          theta2=1/6*360,
                          color=color)

    outer = patches.Ellipse(center,
                             segment_r*4,
                             segment_r*4,
                             fill=False,
                             linewidth=2,
                             color=color)
    ax.add_patch(outer)
    ax.add_patch(apex_b1)
    ax.add_patch(LAD_mid1)
    ax.add_patch(LAD_mid2)
    ax.add_patch(LCX_mid1)

    #RCA/LAD outer
    ax.plot([center[0]-2*segment_r, center[0]-segment_r], [center[0], center[0]],'-', color=color, linewidth=2)

    #LAD/LCX outer
    ax.plot([center[0]+np.cos(-1/6*np.pi*2)*segment_r,
             center[0]+np.cos(-1/6*np.pi*2)*2*segment_r], 
            [center[0]+np.sin(-1/6*np.pi*2)*segment_r, 
             center[0]+np.sin(-1/6*np.pi*2)*2*segment_r],
            '-', color=color, linewidth=2)
    #LAD/LCX inner
    ax.plot([center[0]+np.cos(-1/8*np.pi*2)*.5*segment_r,
             center[0]+np.cos(-1/8*np.pi*2)*segment_r], 
            [center[0]+np.sin(-1/8*np.pi*2)*.5*segment_r, 
             center[0]+np.sin(-1/8*np.pi*2)*segment_r],
            '-', color=color, linewidth=2)

    #RCA/LCX outer
    ax.plot([center[0]+np.cos(1/6*np.pi*2)*segment_r,
             center[0]+np.cos(1/6*np.pi*2)*2*segment_r], 
            [center[0]+np.sin(1/6*np.pi*2)*segment_r, 
             center[0]+np.sin(1/6*np.pi*2)*2*segment_r],
            '-', color=color, linewidth=2)

    #RCA/LCX inner
    ax.plot([center[0]+np.cos(1/8*np.pi*2)*.5*segment_r,
             center[0]+np.cos(1/8*np.pi*2)*segment_r], 
            [center[0]+np.sin(1/8*np.pi*2)*.5*segment_r, 
             center[0]+np.sin(1/8*np.pi*2)*segment_r],
            '-', color=color, linewidth=2)

    #LAD/RCA inner
    ax.plot([center[0]+np.cos(3/8*np.pi*2)*.5*segment_r,
             center[0]+np.cos(3/8*np.pi*2)*segment_r], 
            [center[0]+np.sin(3/8*np.pi*2)*.5*segment_r, 
             center[0]+np.sin(3/8*np.pi*2)*segment_r],
            '-', color=color, linewidth=2)
    return ax


def display_saliency_maps(smaps, polar_plots, y_loc, probabilities, method=None, k=None, study_nos=None, weighted=False, savedir=None):
    """
    smaps :  dict
        Key is the smethod, value is a numpy array of shape
        (num_examples, H, W, num_channels) containing attention maps
        associated with each example
    polar_plots : np array
        (num_examples, H, W, num_channels) array of corresponding polar maps
    y_loc : pd.Series
        Corresponding localization labels
    probabilities : np.array
        Array of corresponding probabilities
    method : list of str
        List of saliency methods present in smaps dict to display
    k : int
        Optional argument. If K is present, display only the firs k attention
        maps
    study_nos : np.array
       Corresponding study numbers 
    weighted : bool
        Whether the attention map should be weighted by predicted probability.
        Suggest that this is set to False. 

    For each included example, this function displays rest and stress polar
    maps along with an attention map. It also shows the study number associated
    with each example, ground truth labels for scar and ischemia in each vessel
    territory, and the model's predicted probability that the example is abnormal. 

    """

    stop = k if k else len(polar_plots)
    if stop > len(polar_plots):
        stop = len(polar_plots)
    
    for i in range(stop):
        if study_nos is not None:
            print('study no:', study_nos[i])
        image = polar_plots[i]
        data = np.array([image])

        display(pd.DataFrame(y_loc.iloc[i]).transpose())

        channels = ['stress', 'rest']
        if method:
            cols = len(channels) + 1
        else:
            cols = len(channels)
        fig, axs = plt.subplots(1, cols, figsize=(cols*4 , 3))


        for channel_i, channel in enumerate(channels):
            cbar=True

            sns.heatmap(
                image[:,:,channel_i],
                ax=axs[channel_i],
                cbar=cbar,
                square=True)
            axs[channel_i] = plot_vessel_boundries(axs[channel_i], rings=20)
            axs[channel_i].axes.set_xticks([])
            axs[channel_i].axes.set_yticks([])

            axs[channel_i].set_title(channel.capitalize())
        plt.plot()

        print('Probability of Abnormal:', probabilities[i])
        vis_style = 'color'
        
        if weighted:
            prob=probabilities[i]
        else:
            prob=None
        
        if method is None:
            num_rows, num_cols = 2, 4
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.5, num_rows * 2))
            for row in range(num_rows):
                for col in range(num_cols):
                    axs[row][col].axes.set_xticks([])
                    axs[row][col].axes.set_yticks([])


            methods = ['BG', 'BG_2', 'BG_VAR', 'BG_STD', 'SH', 'SG', 'SG_2', 'SH_VAR']
            titles = ['BayesGrad', 'BayesGrad 2', 'BayesGrad VAR', 'BayesGrad STD',
                      'VanillaGrad', 'SmoothGrad', 'SmoothGrad 2', 'SmoothGrad VAR']


            for j, method_i in enumerate(methods):
                col = 0 if j < 4 else 1
                row = j % 4
            
                vu.visualize_smap(smaps[method_i][i], ax=axs[col][row], title=titles[j], style=vis_style, prob=prob)
                axs[col][row] = plot_vessel_boundries(axs[col][row], rings=20)

            if savedir:
                plt.savefig(f'{savedir}/ploar_plot_study_no_{study_nos[i]}.png',
                    dpi=300)
            plt.show()
        else:
            axs[2].axes.set_xticks([])
            axs[2].axes.set_yticks([])


            methods = ['BG', 'BG_2', 'BG_VAR', 'BG_STD', 'SH', 'SG', 'SG_2', 'SH_VAR']
            titles = ['BayesGrad', 'BayesGrad 2', 'BayesGrad VAR', 'BayesGrad STD',
                      'VanillaGrad', 'SmoothGrad', 'SmoothGrad 2', 'SmoothGrad VAR']
            title = titles[methods.index(method)]
            vu.visualize_smap(smaps[method][i], ax=axs[2], title=title, style=vis_style, prob=prob)
            ax = plot_vessel_boundries(axs[2], rings=20)

            fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='viridis'), ax=axs[2])
            if savedir:
                plt.savefig(f'{savedir}/ploar_plot_study_no_{study_nos[i]}.png',
                    dpi=300)
            plt.show()



def get_reg_encoding():
    """
    
    """
    reg_encoding = {
        'lad': lambda x: (x['scar_lad'] | x['ischemia_lad']).astype(bool).values,
        'rca': lambda x: (x['scar_rca'] | x['ischemia_rca']).astype(bool).values,
        'lcx': lambda x: (x['scar_lcx'] | x['ischemia_lcx']).astype(bool).values}

    # A dictionary at key region or region group, returns boolean list of the
    # inxexes for the studies that fall in that group when passed the 6 digit
    # string encoding of
    # [scar_lad, scar_rca, scar_lcx, ischemia_lad, ischemia_rca, ischemia_lcx]
    # where the digit is 1 if the value of the label is true else 0
    reg_encoding_full = {
        'lad': lambda x: (reg_encoding['lad'](x) &
                          ~reg_encoding['rca'](x) &
                          ~reg_encoding['lcx'](x)),
        'rca': lambda x: (reg_encoding['rca'](x) &
                          ~reg_encoding['lad'](x) &
                          ~reg_encoding['lcx'](x)),
        'lcx': lambda x: (reg_encoding['lcx'](x) &
                          ~reg_encoding['rca'](x) &
                          ~reg_encoding['lad'](x)),
        'normal': lambda x: (~reg_encoding['lad'](x) &
                             ~reg_encoding['rca'](x) &
                             ~reg_encoding['lcx'](x)),
        'lad_rca': lambda x: (reg_encoding['lad'](x) &
                              reg_encoding['rca'](x) &
                              ~reg_encoding['lcx'](x)),
        'lad_lcx': lambda x: (reg_encoding['lad'](x) &
                              reg_encoding['lcx'](x) &
                              ~reg_encoding['rca'](x)),
        'rca_lcx': lambda x: (reg_encoding['rca'](x) &
                              reg_encoding['lcx'](x) &
                              ~reg_encoding['lad'](x)),
        'lad_rca_lcx': lambda x: (reg_encoding['lad'](x) &
                                  reg_encoding['rca'](x) &
                                  reg_encoding['lcx'](x))}
    
    return reg_encoding_full



def plot_avg_maps(avg_maps,  normalized=True, title_prefix=None, counts=None, savepath=None): 
    """
    avg_maps : nested python dictionary
        This is the nested dictionary containing averaged attention maps
        for every possible regional distribution of disease -- this is the
        output of mean_maps.

    normalized : bool
        Whether or not to show each mean attention maps on a normalized scale
        between 0 and 1

    title_prefix : str
        Optional prefix to add to plot title

    counts : dictionary
        Optional. Counts is output of mean_maps with counts for the number of 
        examples corresponding to each possible regional distribution of disease.
        If this is passed, this is used in the titles of each individual
        mean smap.

    savepath : str
        Optional path for where plot will be saved. Plot is not saved if no
        path is passed. 
    """
    reg_encoding_full = get_reg_encoding()
    methods = list(avg_maps.keys())
    
    for method in methods:
        rest_or_stress = (method == 'stress') or (method == 'rest')
        fig, axs = plt.subplots(2, 4, figsize=(12 * 1.5, 6))
        if title_prefix:
            title = f"{title_prefix} {method.upper()}"
        else:
            title = method.upper()

        plt.suptitle(title)
        
        vmax = np.NINF
        # Need to make sure all on same scale if use diff
        for i, reg in enumerate(reg_encoding_full.keys()):
            reg_max = np.abs(avg_maps[method][reg]).max()
            if reg_max > vmax:
                vmax = reg_max
            
        vmin = -vmax        
        for i, reg in enumerate(reg_encoding_full.keys()):
            if i >3:
                row=1
            else:
                row=0

            if i > 3:
                col = i - 4
            else:
                col = i
                
            if counts:
                title = f"{reg} (n={counts[reg]})"
            else:
                title = reg

            if rest_or_stress:
                sns.heatmap(
                avg_maps[method][reg],
                ax=axs[row][col],
                cbar=True,
                square=True)
                axs[row][col] = plot_vessel_boundries(axs[row][col], rings=20)
                axs[row][col] = plot_vessel_boundries(axs[row][col], rings=20)
                axs[row][col].set_title(title)
            else:
                if not normalized:
                    vu.visualize_smap(avg_maps[method][reg], ax=axs[row][col], title=title, style='color')
                    axs[row][col] = plot_vessel_boundries(axs[row][col], rings=20)
                else:
                    plot = axs[row][col].imshow(avg_maps[method][reg], vmin=0, vmax=vmax, cmap='viridis')
                    axs[row][col].set_title(title)
                    axs[row][col] = plot_vessel_boundries(axs[row][col], rings=20)

            axs[row][col].axes.set_xticks([])
            axs[row][col].axes.set_yticks([])


        if not rest_or_stress:
            if normalized:
                norm = colors.Normalize(vmin=0, vmax=vmax)
                fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axs)
            else:
                fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='viridis'), ax=axs)

        if savepath is not None:
            plt.savefig(savepath, dpi=300)
        plt.plot()


def mean_maps(maps, y_loc, methods, probabilities=None, counts=False):
    """
    maps: dict
        key is method, val is vector of maps associatd with that method.
        Valid methods include all saliency methods if taking the mean of
        attention maps, or 'rest' or 'stress' if taking the mean of raw
        polar map data.
    
    y_loc: pd.dataframe
        labels
        
    methods: list
        methods to plot

    probabilities: np.array
        Predicted probabilities coresponding to saliency maps. Used for to
        generate weighted average of the saliency maps. Optional, if not
        provided, the average will not be weighted.

    counts: bool
        If True, return dictionary with population counts for each region

    This function returns:
        avg_smaps : nested python dictionary
            Each key in the python dictionary is saliency method, 'rest' or
            'stress'. The value corresponding to this key is a second python
            dictionary containing keys corresponding to every combination of
            vessel territory as well as 'normal' to represent every possible
            combination of obstructive CAD. The values in this dictionary are
            an attention map (or polar map if methods include 'rest' or
            'stresss') which represents the average of all attention maps (or
            polar maps) for all patients in the input who have disease in the
            given combination of vessels. 

        counts (optional) : dict
            This dictionary has keys that match those of the inner dictionaries
            in avg_smaps, and the values represent the number of examples in
            the data which have abnormality that matches the given combination
            of abnormal vessels.
    """
    avg_maps = defaultdict(dict)
    
    
    counts_by_reg = {}
    for method_i, method in enumerate(methods):
        method_i_maps =  np.array(maps[method])

        reg_encoding_full = get_reg_encoding()

        for reg in reg_encoding_full.keys():            
            maps_normed = []

            reg_mask = reg_encoding_full[reg](y_loc)
            counts_by_reg[reg] = reg_mask.astype(int).sum()
            
            # Iterate over list of all maps that have given presentation
            for smap in method_i_maps[reg_mask]:
                maps_normed.append(vu.abs_grayscale_norm(smap))
                
            maps_normed = np.array(maps_normed)
            
            if probabilities is not None:

                
                if reg == 'normal':
                    probs = (1 - probabilities)[reg_mask].reshape(-1, 1, 1)
                else:
                    probs = probabilities[reg_mask].reshape(-1, 1, 1)
                    
                avg_maps[method][reg] = (maps_normed * probs).mean(axis=0)
            else:
                avg_maps[method][reg] = maps_normed.mean(axis=0)
            
    if counts:
        return deepcopy(avg_maps), counts_by_reg      
    else:
        return deepcopy(avg_maps)


def confusion_matrix(y_true, y_pred, scores=np.array([None]), axis_labels=None, title=None, filepath=None):
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
    sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=ax)
    if axis_labels:
        ax.set_yticklabels(axis_labels)
        ax.set_xticklabels(axis_labels)
#     ax.set_ylim([0,2])
    if title:
        ax.set_title(title)
    if filepath:
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)


def load_saved_smaps(smethods, split, base_path, model_alias):
    """
    smethods: list of str
        smethods for which to load smaps

    split: str
        train, val, or test. Test is the union of the following splits from
        PET aim 1: validation for model comparison, evaluation for generalization
        performance.

    base_path: str
        Path to directory containing directories of saved attention maps for
        various models. Output_dir from saliency_map_generator.py

    model_alias: str
        Directory within base_path to identify specific model/model instantiation.
        This should match what was used in saliency_map_generator.py

    path to saliency maps for smethod should be in format:
    {base_path}/{model_alias}/{split}/{smethod}.json . This is the file structure
    created by saliency_map_generator.py

    returns dictionary {'s_method': dict(smethod_1: np.array,
                                    ...,
                                    smethod_i: np.array),
                         'study_nos': study_nos,
                         'peobabilities': probabilities)}

    """

    def load_smaps(smethods, split_dir, base_path, model_alias):

        base_path = os.path.join(base_path, model_alias)
        s_maps = {}
        for smethod in smethods:
            file_path = f'{base_path}/{split_dir}/{smethod}.json'
            with open(file_path, "r") as infile:
                smaps_loaded = json.load(infile)
            s_maps[smethod] = np.array(smaps_loaded)

        file_path = os.path.join(
            base_path,
            split_dir,
            'study_nos.csv')

        study_nos = pd.read_csv(file_path).values.reshape(-1)

        file_path =  os.path.join(
            base_path,
            split_dir,
            'probabilities.csv')
        probabilities = pd.read_csv(file_path).values.reshape(-1)

        return s_maps, study_nos, probabilities
    
    
    if split != 'test':
        if split == 'validation':
            split_dir = 'validation_hp'
        else:
            split_dir = 'train'


        s_maps, study_nos, probabilities = load_smaps(smethods, split_dir, base_path, model_alias)

    else:
        s_maps_val, study_nos_val, probabilities_val = load_smaps(smethods, 'validation_mc', base_path, model_alias)
        s_maps_test, study_nos_test, probabilities_test = load_smaps(smethods, 'test', base_path, model_alias)

        study_nos = np.hstack((study_nos_val, study_nos_test))
        probabilities = np.hstack((probabilities_val, probabilities_test))

        s_maps = {}
        for method in smethods:
            s_maps[method] = np.vstack((s_maps_val[method], s_maps_test[method]))

    return {'s_maps': s_maps,
            'study_nos': study_nos,
            'probabilities': probabilities}



def load_data_for_smaps(split, data_path):
    """
    split : str
        Valid options include 'train', 'validation', and 'test'

    data_path : str
        Path to directory holding the datasets from PET aim 1 analysis.  
    
    Load data with localization labels for the given split. If split is 'train'
    return subset of training split where val_split is 0 when using nn_val_split.
    If split is 'validation', return subset of training split where val_split is
    1 when using nn_val_split. If split is 'test', return concationation of 
    validation set for model comparison from PET project aim 1 with test
    set for generalizability from aim 1. In PET aim 1, the validation split for 
    model comparison was used to compare performance between different models 
    after hyperparameters were selected, while the test set for generalizability
    was used to test generalization performance of the highest performing model.
    Because there are few models compared in this analysis and because
    generalization performance was not the main outcome investigated, this 
    analysis uses a single test set (the union of the two previously described).
    
    """
    if split != 'test':
        if split == 'validation':
            split_dir = 'validation_hp'
        else:
            split_dir = 'train'
            
        data_loc = dup.load_dataset(data_path, 'polar_plot', 'localization', 'train', val_col='nn_val_split')

        y_loc = data_loc['y']
        if split == 'validation':
            y_loc = y_loc[data_loc['val_split'] == 1]
            X = data_loc['X'][data_loc['val_split'] == 1]
        elif split == 'train':
            y_loc = y_loc[data_loc['val_split'] == 0]
            X = data_loc['X'][data_loc['val_split'] == 0]

        y_loc = y_loc.reset_index(drop=True)
        X = X[:,:,:,:2]

    else:
        val = dup.load_dataset(data_path, 'polar_plot', 'localization', 'val')
        test = dup.load_dataset(data_path, 'polar_plot', 'localization', 'test')

        y_loc = pd.concat((val['y'], test['y']))
        X = np.vstack((val['X'], test['X']))
        # just rest and stress
        X = X[:,:,:,:2]
        
    return X, y_loc



