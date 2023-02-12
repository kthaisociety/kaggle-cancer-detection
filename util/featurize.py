import numpy as np
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, moments, moments_hu


def extract_normalized_features(image, segments):
    """
    This function takes an image and segments as inputs and returns the features of the segments normalized using MinMaxScaler.
    
    Parameters:
    image (numpy.ndarray): Input image
    segments (numpy.ndarray): Image segments
    
    Returns:
    list: List of dictionaries where each dictionary contains the normalized features for one segment
    
    """
    
    features = extract_features(image, segments)
    return scale_dict_list(features)

def extract_features(image, segments, include_background=False):
    
    """
    This function extracts various features from a given image and its corresponding segments.

    Parameters:
    image (ndarray): The input image.
    segments (ndarray): The segments of the input image.
    include_background (boolean): whether to include the background segment, default to False

    Returns:
    list: A list of dictionaries, where each dictionary contains the extracted features for one superpixel in the input image.
    The extracted features include:
    - 'area': the area of the superpixel
    - 'perimeter': the perimeter of the superpixel
    - 'circularity': the circularity of the superpixel
    - 'eccentricity': the eccentricity of the superpixel
    - 'solidity': the solidity of the superpixel
    - 'intensity_mean': the mean intensity of the superpixel
    - 'intensity_std': the standard deviation of the intensity of the superpixel
    - 'lbp_mean': the mean of the local binary pattern of the superpixel
    - 'lbp_std': the standard deviation of the local binary pattern of the superpixel
    - 'hu_mean': the mean of the Hu moments of the superpixel
    - 'hu_std': the standard deviation of the Hu moments of the superpixel
    - 'hu1': the first Hu moment of the superpixel
    - 'hu2': the second Hu moment of the superpixel
    - 'hu3': the third Hu moment of the superpixel
    - 'hu4': the fourth Hu moment of the superpixel
    - 'hu5': the fifth Hu moment of the superpixel
    - 'center_x': the x coordinate of the center of the superpixel
    - 'center_y': the y coordinate of the center of the superpixel
    """
    
    segments_ids = np.unique(segments)
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    props = regionprops(segments, intensity_image=image) # this thing is removing the last segment for some reason
    superpixel_features = []
    for i, prop in enumerate(props):
        labels = prop.label
        binary_image = np.zeros_like(segments)
        binary_image[segments == labels] = 1
        intensity_image = image * binary_image
        threshold = threshold_otsu(intensity_image)
        binary_image = intensity_image > threshold

        area = prop.area
        perimeter = prop.perimeter
        circularity = 4 * np.pi * area / (perimeter**2)
        eccentricity = prop.eccentricity
        solidity = prop.solidity
        intensity_mean = prop.mean_intensity
        intensity_std = prop.intensity_image.std()
        lbp = local_binary_pattern(img_as_ubyte(intensity_image), 8, 1)
        lbp_hist = np.histogram(lbp, bins=range(257))[0]
        
        lbp_mean = lbp_hist.mean()
        lbp_std = lbp_hist.std()

        m = moments(binary_image)
        hu_moments = moments_hu(m)
        hu1, hu2, hu3, hu4, hu5, hu6, hu7 = hu_moments
        hu_mean = np.mean(hu_moments)
        hu_std = np.std(hu_moments)

        superpixel_features.append({
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
            'hu_mean': hu_mean,
            'hu_std': hu_std,
            'hu1': hu1,
            'hu2': hu2,
            'hu3': hu3,
            'hu4': hu4,
            'hu5': hu5,
            'center_x': centers[i][0],
            'center_y': centers[i][1]
            
        })
    # TODO one node is lost here, the code removes the biggest node, which is not the entire background, when it is divided in smaller pices
    # if not include_background:
    #     # skip the node with the biggest area (background):
    #     areas = [superpixel_features[i]['area'] for i in range(len(superpixel_features))]
    #     background_feature_set_index = areas.index(max(areas))
    #     del superpixel_features[background_feature_set_index]

    return superpixel_features


def scale_dict_list(dict_list):
    """
    Scale the values in each dictionary of the list using MinMax Scaler method.
    
    The function takes in a list of dictionaries and scales the values for each key in each dictionary. 
    The scaling is done based on the values across all dictionaries in the list, not within each individual dictionary.
    
    Parameters:
    dict_list (list of dictionaries): A list of dictionaries where each dictionary contains key-value pairs to be scaled.
    
    Returns:
    list of dictionaries: A list of dictionaries with scaled values for each key.
    
    Example:
    >>> dict_list = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    >>> scale_dict_list(dict_list)
    [{'a': 0.0, 'b': 0.0}, {'a': 1.0, 'b': 1.0}]
    
    """
    # Import the MinMaxScaler module from scikit-learn library
    from sklearn.preprocessing import MinMaxScaler
    
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Extract the values from each dictionary in the list and create a 2D numpy array
    dict_values = [list(d.values()) for d in dict_list]
    data = np.array(dict_values)
    
    # Fit the scaler to the data and perform scaling
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    
    # Convert the scaled data back to a list of dictionaries
    scaled_dict_list = []
    for i, d in enumerate(dict_list):
        scaled_dict = {}
        for j, key in enumerate(d.keys()):
            scaled_dict[key] = scaled_data[i, j]
        scaled_dict_list.append(scaled_dict)
    
    return scaled_dict_list


    