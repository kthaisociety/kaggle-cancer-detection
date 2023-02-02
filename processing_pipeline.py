import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from segmentation import segment_img
from featurize import extract_normalized_features
from graph_construction import generate_graph, save_graph

GRAPH_DATA_DIRNAME = 'graphs'

if __name__ == "__main__":
    
    # create the directory to hold graph data
    if not os.path.isdir(GRAPH_DATA_DIRNAME):
        os.mkdir(GRAPH_DATA_DIRNAME)

    images = [plt.imread("64956_1305773827.png")] # placeholder, numpy arrays for each image

    for image in images:
        
        # segment the image
        segments = segment_img(image)
        
        # get features from each segment
        features = extract_normalized_features(image, segments)
        
        # construct the graph
        graph = generate_graph(features)
        
        # save the graph
        save_graph(graph, f"{GRAPH_DATA_DIRNAME}/a_really_cool_graph_name.pkl")
    