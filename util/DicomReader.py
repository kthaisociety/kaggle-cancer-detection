import pydicom
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

class DicomReader:
    """
    Utility class to handle DICOM files
    Initialize with .csv file containing information about .dcm files
    TBA
    """
    
    def __init__(self, table_path) -> None:
        """
        initialize dicomreader class
         Parameters
        ----------
        table_path : str
            path to table.csv file

        Returns
        -------
        """
        self.table =  pd.read_csv(table_path, dtype={'patient_id':str, 'image_id':str})

    def extract_img_and_meta(self, file_path, scale_ratio=1, plot_img=True):
        """
        extracts the pixel data and cancer data and resizes image
        Parameters
        ----------
        file_path : str
            path to .dcm file
        scale_ratio : float
            scaling ratio for resizing, default 1 -> not scaling
        plot_img : bool
            whether to plot img or not
        
        Returns
        -------
            patient_id: str  
                id of patien
            image_id : str 
                id of image
            img : ndarray
                pixel data as array
            cancer: bool 
                binary indicator whether patient has cancer or not
        """   
        patient_id = file_path.split('/')[-2]
        image_id = file_path.split('/')[-1][:-4]

        cancer = self.table[(self.table["patient_id"]==patient_id) & (self.table["image_id"]==image_id)].cancer.iloc[0]

        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array

        width = int(img.shape[1] * scale_ratio)
        height = int(img.shape[0] * scale_ratio)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        img = (img - img.min()) / (img.max() - img.min())

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
        
        if plot_img:
            plt.figure(figsize=(5, 5))
            plt.imshow(img, cmap="gray")
            plt.title(f"{patient_id} {image_id}")
            plt.show()

        return patient_id, image_id, img, cancer


    def process_data(self, f, size=512, save_folder="", extension="png"):
        """
        """
        patient = f.split('/')[-2]
        image = f.split('/')[-1][:-4]

        dicom = pydicom.dcmread(f)
        img = dicom.pixel_array

        img = (img - img.min()) / (img.max() - img.min())

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img

        img = cv2.resize(img, (size, size))

        cv2.imwrite(save_folder + f"{patient}_{image}.{extension}", (img * 255).astype(np.uint8))


if __name__=="__main__":
    dr = DicomReader("toy_data/train.csv")
    patient_id, image_id, img, cancer = dr.extract_img_and_meta("toy_data/train_images/30699/961718628.dcm", scale_ratio=.1, plot_img=True)
    print(np.max(img))
    #dr.process_data("data/train/10006/462822612.dcm", "data/train.csv")