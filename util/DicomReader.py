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

    def extract_img_and_meta(self, file_path, plot_img=True):
        """
        extracts the pixel data and cancer data
        Parameters
        ----------
        file_path : str
            path to .dcm file
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
    dr = DicomReader("data/train.csv")
    print(dr.extract_img_and_meta("data/train/24947/1231101161.dcm"))
    #dr.process_data("data/train/10006/462822612.dcm", "data/train.csv")