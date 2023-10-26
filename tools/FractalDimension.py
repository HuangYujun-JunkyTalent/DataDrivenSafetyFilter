#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:47:15 2019

@author: daniel

Modified by Yujun Huang, to deal with general shape/non-integer samples
"""
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

def fractal_dimension(
        locs: np.ndarray, region_min: np.ndarray, region_max: np.ndarray,
        max_box_size: float = None, min_box_size: float = -3.0,
        n_samples: int = 20, n_offsets: float = 0, plot = False) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculates the fractal dimension of a 3D numpy array.
    Returns: 
        float: the fractal dimension
        np.ndarray: the log of the number of boxes N(s) needed to cover the graph
        np.ndarray: the log of the inverse of the box size 1/s

    Args:
        locs (np.ndarray): locations where the graph is marked as 1. 
                            Of the form [[point1], [point2], ...]
        region_min (np.ndarray): minimum values of the region of interest.
        region_max (np.ndarray): maximum values of the region of interest.
        max_box_size (float): The largest box size, given as the power of 2 so that
                            2**max_box_size gives the sidelength of the largest box.                     
        min_box_size (float): The smallest box size, given as the power of 2 so that
                            2**min_box_size gives the sidelength of the smallest box.
                            Default value 1.
        n_samples (int): number of scales to measure over.
        n_offsets (float): number of offsets to search over to find the smallest set N(s) to
                       cover  all voxels>0.
        plot (bool): set to true to see the analytical plot of a calculation.
                            
        
    """
    #determine the scales to measure on
    if max_box_size == None:
        #default max size is the largest power of 2 that fits in the smallest dimension of the array:
        max_box_size = np.log2(np.min(region_max - region_min))
    scales = np.logspace(max_box_size, min_box_size, num=n_samples, base=2)
    
    #count the minimum amount of boxes touched
    Ns = []
    #loop over all scales
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        #search over all offsets
        for offset in offsets:
            bin_edges = [np.arange(min_el-offset, max_el+scale, scale) for min_el, max_el in zip(region_min, region_max)]
            H1, e = np.histogramdd(locs, bins = bin_edges)
            touched.append(np.sum(H1>0))
        Ns.append(touched)
    Ns = np.array(Ns)
    
    #From all sets N found, keep the smallest one at each scale
    Ns = Ns.min(axis=1)
   
    
    #Only keep scales at which Ns changed
    scales  = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])
    
    
    Ns = np.unique(Ns)
    Ns = Ns[Ns > 0]
    scales = scales[:len(Ns)]
    #perform fit
    coeffs = np.polyfit(np.log2(1/scales), np.log2(Ns),1)
    
    #make plot
    if plot:
        fig, ax = plt.subplots(figsize = (8,6))
        ax.scatter(np.log2(1/scales), np.log2(np.unique(Ns)), c = "teal", label = "Measured ratios")
        ax.set_ylabel(r"$\log_2 N(\epsilon)$")
        ax.set_xlabel(r"$\log_2 1/ \epsilon$")
        # fitted_y_vals = np.polyval(coeffs, np.log2(1/scales))
        # ax.plot(np.log2(1/scales), fitted_y_vals, "k--", label = f"Fit: {np.round(coeffs[0],3)}X+{coeffs[1]}")
        # ax.legend()
    
    return coeffs[0], np.log2(Ns), np.log2(1/scales)


def plot_squares(locs_list: List[np.ndarray], scale: float, region_min: np.ndarray, region_max: np.ndarray, n_offsets: float = 0, ax = None):
    """Plots the squares used to cover all the locs as in fractal_dimension method.
    Args:
        locs_list (List[np.ndarray]): list of locations where the graph is marked as 1.
        scale: scale of the box size, in log2 scale. 
        region_min (np.ndarray): minimum values of the region of interest.
        region_max (np.ndarray): maximum values of the region of interest.
        n_offsets: use this many offsets to find the smallest set N(s) to cover all locs.
        ax (matplotlib.axes.Axes): axes to plot on. If None, a new figure is created.
    """
    if ax == None:
        fig, ax = plt.subplots(figsize = (8,6))
    for locs in locs_list:
        ax.plot(locs[:,0], locs[:,1], color = "tab:blue")
    ax.set_xlim(region_min[0], region_max[0])
    ax.set_ylim(region_min[1], region_max[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    all_locs = np.vstack(locs_list)
    