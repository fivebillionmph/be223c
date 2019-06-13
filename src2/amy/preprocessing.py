# -*- coding: utf-8 -*-
"""
Author: Amy Cummings

This module provides directory set up, creates png files from 

Example:
    $ python preprocessing.py /path/to/directory/

Attributes:
    
    folders: 
    
    get_pixels_hu: 

Todo:
    1. Run command, which creates folders.
    
    2. Place original images in 'Images/' folder. Note if automated segmentation
       is not desired, manually created masks may be placed in 'Preproc/' folder.
    
    3. Create CSV with names of images files without extension in first column
       and binary outcome variable in second.
       
       Example:
            A012345,1
            
    3. Hash folder command and unhash remaining functions, rerun script. Note if 
       manually created masks are provided, only preproc function should be unhashed.
    
    
"""

# packages

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

# sets up folders that will be used

def folders():
    os.makedirs('Images/')
    os.makedirs('
              
# uses metadata from dicom file to convert pixels to hounsfield units

def get_pixels_hu(data, slope, intercept):
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

# uses hu values to make lungmask and perform segmentation

def make_lungmask(img, display=False):
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
    
    return final

# runs preprocessing, sets paths

ospath = os.getcwd()
os.chdir('/path/')    

path = "Images/"
csv = "Master.csv"
seg = "Seg/"
proc = "Preproc/"


# uses filenames to define paths, requires filenames in first column without file extensions, assumed dicom images

df = pd.read_csv(csv, header=0)
X = df.iloc[0:1,0]
    
# runs loop

for filename in X:
    I = path + filename + '.dcm'
    P = proc + filename + '.dcm'
    S = seg + filename + '.png'
    image = [I]
    process = [P]
    lungs = [S]
    ds = dicom.dcmread(I, force=True)
    data = ds.pixel_array
    slope = ds.RescaleSlope
    intercept = ds.RescaleIntercept



# shows and saves output

    plt.imshow(final)
    im = Image.fromarray(final*128)
    im = im.convert("L")
    im.save(S)
    
    lungs = final[::2, ::2]
    ds.PixelData = lungs.tobytes()
    ds.Rows, ds.Columns = lungs.shape
    ds.save_as(P)

# runs code, 
    
folders()
#preproc()
#makepng()
#img = get_pixels_hu(data, slope, intercept)
#final = make_lungmask(img)
