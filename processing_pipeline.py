import os

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from util.segmentation import segment_img
from util.featurize import extract_normalized_features
from graph_construction import generate_graph, save_graph
from util.DicomReader import DicomReader

GRAPH_DATA_DIRNAME = 'graphs'

if __name__ == "__main__":

    SCALE_RATIO = .25
    
    # create the directory to hold graph data
    if not os.path.isdir(GRAPH_DATA_DIRNAME):
        os.mkdir(GRAPH_DATA_DIRNAME)

    image_reader = DicomReader("toy_data/train.csv")
    patient_id, image_id, img, cancer = image_reader.extract_img_and_meta("toy_data/train_images/24947/1231101161.dcm", plot_img=False)

    width = int(img.shape[1] * SCALE_RATIO)
    height = int(img.shape[0] * SCALE_RATIO)
    dim = (width, height)
    print(f"resizing image from: ({img.shape[1]}, {img.shape[0]}) to ({width},{height})")
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    images = [img] # placeholder, numpy arrays for each image

    for image in images:
        
        # segment the image
        print(f"Patien: {patient_id}, Image: {image_id}, Cancer: {cancer}\n----------------------------------------------------")
        print(f"segmenting")
        segments = segment_img(image)
        
        # get features from each segment
        print(f"featurizing")
        features = extract_normalized_features(image, segments)
        
        # construct the graph
        print("generating graph")
        graph = generate_graph(features)
        
        # save the graph
        file_name = f"{patient_id}_{image_id}"
        file_name += "_cancer" if cancer else "_nocancer"
        save_graph(graph, f"{GRAPH_DATA_DIRNAME}/{file_name}.pkl")
        print(f"----------------------------------------------------")
    