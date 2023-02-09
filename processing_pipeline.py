import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

from util.segmentation import segment_img
from util.featurize import extract_normalized_features
from graph_construction import generate_graph, save_graph
from util.DicomReader import DicomReader

GRAPH_DATA_DIRNAME = 'graphs'

if __name__ == "__main__":
    
    # create the directory to hold graph data
    if not os.path.isdir(GRAPH_DATA_DIRNAME):
        os.mkdir(GRAPH_DATA_DIRNAME)

    image_reader = DicomReader("toy_data/train.csv")
    patient_id, image_id, img, cancer = image_reader.extract_img_and_meta("toy_data/train_images/24947/1231101161.dcm", plot_img=False)
    images = [img] # placeholder, numpy arrays for each image

    for image in images:
        
        # segment the image
        print(f"{patient_id} {image_id}\n ----------------------------------------------------")
        print(f"segmenting")
        segments = segment_img(image)
        
        # get features from each segment
        print(f"featurizing")
        features = extract_normalized_features(image, segments)
        
        # construct the graph
        print("generating graph")
        graph = generate_graph(features)
        
        # save the graph
        file_name = "cancer" if cancer else "no_cancer"
        save_graph(graph, f"{GRAPH_DATA_DIRNAME}/{file_name}.pkl")
    