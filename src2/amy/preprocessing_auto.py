# -*- coding: utf-8 -*-
"""
Author: Amy Cummings
Language: python v3.6+
Required packages: 
    matplotlib, numpy, pandas, PIL, pydicom, sklearn, skimage
    
    
This script provides directory set up and automated segmentation using dicom files. 
Note first run creates folders, after which ''' must be removed and the folder module 
hashed to run automated segmentation. There is a separate script for manual segmentation 
support (preprocessing_man.py) that can be substituted for this script.

Example:
    $ python preprocessing.py /path/to/directory/


Attributes:
    
    folders(): creates folders that the program will call in the directory folder. Note 
    images need to placed into appropriate folders after they are created (see Todo).

     
    preproc(): extracts information from dicom files to enable conversion of image into
    Housfield units. This requires a CSV entitled "Master.csv" placed in 
    /path/to/directory/folder with image names (see Todo). 

        Returns:
            
            data: this is an extracted pixel array from the dicom file.
            
            slope: this is a value extracted from the dicom file used to calculate
            Hounsfield units.
            
            intercept: this is a value extracted from the dicom file used to calculate
            Hounsfield units.
     
     
    get_pixels_hu(data, slope, intercept): uses extracted information from dicom to 
    calculate Hounsfield units that are used for automated segmentation.
    
        Args:
        
            data: pixel array from dicom image. 
            
            slope: value extracted from dicom metadata used to modify pixel array based on
            Hounsfield units.
            
            intercept: value extracted from dicom metadata used to modify pixel array based on
            Hounsfield units.
        
        Returns:
     
            img: this is a np array based on Hounsfield units used to create the lungmask.
    
    
    make_lungmask(img): Uses k-means and erosion/dilation to create lungmask and then provide
    lung segmentation. Segmentation is visualized in pyplot and saved as a png with same filename 
    as the original dicom file in Seg/.
    
        Args:
        
            img: np array that is used to create lungmask.
        

Todo:

    1. Run command, which creates folders.
    
    2. Place original dicom files in 'Images/' folder.
    
    3. Create CSV with names of images files without extension in first column
       and binary outcome variable in second.
       
       Example:
            A012345,1
            
    4. Remove ''' in the call function section and hash folder() in line 195.
    
    5. Rerun script.
    
"""

""" packages """

import os
import sys
import numpy as np
import pandas as pd
import pydicom as dicom
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt
from PIL import Image


def folders():
    """ sets up folders that will be used """
    os.makedirs('Images/')
    os.makedirs('Seg/')

    return


def preproc():
    ds = dicom.dcmread(I, force=True)
    data = ds.pixel_array
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept                      
    
    return data, slope, intercept


def get_pixels_hu(data, slope, intercept):
    """ uses metadata from dicom file to convert pixels to hounsfield units """
    image = data.astype(np.int16)
    image[image == -2000] = 0
    intercept = intercept
    slope = slope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    img = np.array(image, dtype=np.int16)
    
    return img


def make_lungmask(img, display=False):
    """ uses hu values to make lungmask and perform segmentation """
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    # uses hounsfield values near lungs to normalize images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean
    
    # uses kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)

    # performs erosion and dilation

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    # makes mask

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    final = mask * img
        
    # shows and saves output

    plt.imshow(final)
    im = Image.fromarray(final*128)
    im = im.convert("L")
    im.save(S)
    
    return

""" runs preprocessing, sets paths """

ospath = os.getcwd()
os.chdir(sys.argv[1])    

folders()

'''
""" uses filenames to define paths, requires filenames in first column without file extensions, assumed dicom images """

csv = "Master.csv"
path = "Images/"
seg = "Seg/"

df = pd.read_csv(csv, header=0)
X = df.iloc[:,0]
    
""" runs loop """

for filename in X:
    I = path + filename + '.dcm'
    S = seg + filename + '.png'

    preproc()
    img = get_pixels_hu(data, slope, intercept)
    make_lungmask(img)
'''
