import os

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from util.segmentation import segment_img
from util.featurize import extract_normalized_features
from graph_construction import generate_graph, save_graph
from util.DicomReader import DicomReader



if __name__ == "__main__":

    SCALE_RATIOS = [.25, .5]

    for SCALE_RATIO in SCALE_RATIOS:

        GRAPH_DATA_DIRNAME = f'graphs/scale_ratio_{SCALE_RATIO}'

        DATA_FOLDER = "toy_data/train_images/"

        LOG_FILE = "log.txt"
        
        # create the directory to hold graph data
        if not os.path.isdir(GRAPH_DATA_DIRNAME):
            os.mkdir(GRAPH_DATA_DIRNAME)

        image_reader = DicomReader("toy_data/train.csv")

        for d in os.listdir(DATA_FOLDER):
            for f in os.listdir(f'{DATA_FOLDER}/{d}/'):

                try:
                    patient_id, image_id, img, cancer = image_reader.extract_img_and_meta(f"{DATA_FOLDER}/{d}/{f}", scale_ratio=SCALE_RATIO, plot_img=False)

                    file_name = f"{patient_id}_{image_id}"
                    file_name += "_cancer" if cancer else "_nocancer"

                    if os.path.isfile(f'{GRAPH_DATA_DIRNAME}/{file_name}'+".pkl"):
                        continue
                    
                    images = [img] # placeholder, numpy arrays for each image

                    for image in images:
                        
                        # segment the image
                        print(f"Patient: {patient_id}, Image: {image_id}, Cancer: {cancer}")
                        segments = segment_img(image)
                        
                        # get features from each segment
                        features = extract_normalized_features(image, segments)
                        
                        # construct the graph
                        graph = generate_graph(features)
                        
                        # save the graph
                        
                        save_graph(graph, f"{GRAPH_DATA_DIRNAME}/{file_name}.pkl")
                except Exception as e:
                    with open(LOG_FILE, 'a') as f:
                        f.write(f'{patient_id}_{image_id}: {str(e)}\n')
                        print(f"Error in {patient_id}_{image_id}.")
