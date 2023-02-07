from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import numpy as np


def segment_img(img, scale=100, sigma=0.25, min_size=100):
    """
    Segment an image using the Felzenszwalb's method from skimage.segmentation.

    Parameters
    ----------
    img : ndarray
        Input image to be segmented.
    scale : float, optional
        The spatial resolution of the segmentation. Larger values mean larger segments.
        The default value is 100.
    sigma : float, optional
        Standard deviation of the Gaussian smoothing to apply before running the algorithm.
        The default value is 0.25.
    min_size : int, optional
        The minimum size of segments. Smaller segments will be removed.
        The default value is 100.

    Returns
    -------
    ndarray
        The segmented image, with each segment represented as a unique label.
    """
    
    return felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size, channel_axis=None)



def visualize_segments(img, segments, ax=None):
    
    """
    Visualize the segmentation of an image using matplotlib.

    Parameters
    ----------
    img : ndarray
        Input image to be visualized.
    segments : ndarray
        The segmented image, with each segment represented as a unique label.
    ax : Matplotlib axis, optional
        The Matplotlib axis to plot on. If not provided, a new figure and axis will be created.

    Returns
    -------
    Matplotlib axis
        The axis containing the plotted image.
    """
    
    if not ax:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)

    ax.imshow(mark_boundaries(img, segments)) 

        
    return ax