import pydicom
import matplotlib.pyplot as plt
import cv2
import numpy as np

class DicomReader:
    """
    Utility class to handle DICOM files
    TBA
    """
    
    def __init__(self) -> None:
        pass

    def extract_img_and_meta(self, f):
        """
        """   
        patient_id = f.split('/')[-2]
        image_id = f.split('/')[-1][:-4]

        dicom = pydicom.dcmread(f)
        img = dicom.pixel_array

        img = (img - img.min()) / (img.max() - img.min())

        if dicom.PhotometricInterpretation == "MONOCHROME1":
            img = 1 - img
            
        # plt.figure(figsize=(15, 15))
        # plt.imshow(img, cmap="gray")
        # plt.title(f"{patient} {image}")
        # plt.show()

        return patient_id, image_id, img

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
    dr = DicomReader()
    # dr.extract_img_and_meta("data/train/10006/462822612.dcm")
    dr.process_data("data/train/10006/462822612.dcm")