# -*- coding: utf-8 -*-
"""
Author: Amy Cummings
Language: python v3.6+
Required packages: 
    cv2, pandas, PIL, pydicom


This script provides directory set up and creates pngs from dicom files that can
be used to create masks manually. These masks can then be used for segmentation with
this script. Note first run creates folders, after which ''' must be removed. Functions 
can be called based on hashing (see Todo). There is a separate script with automated 
segmentation (preprocessing_auto.py) that can be substituted for this script.

Example:
    $ python preprocessing_man.py /path/to/directory/folder
    
Todo:

    1. Run command, which creates folders.
    
    2. Place original images in 'Images/' folder.
    
    3. Create CSV with names of images files without extension in first column
       and binary outcome variable in second.
       
       Example:
            A012345,1
            
    4. Masks can be created manually with ImageJ from dicom files. Use the 
       Selection > create mask function, then File > save as > PNG. Name masks with the 
       same filename as the original dicom. Move into Mask/ folder.
    
    5. Remove ''' and hash folder() in line 102, rerun script.
    
"""

# packages

import os
import sys
import pandas as pd
import pydicom as dicom
from PIL import Image
import cv2


# folders module
# sets up folders that will be used - only to be run once then hashed

def folders():
    """
    creates folders that the program will call in the directory folder. Note 
    images need to placed into appropriate folders after they are created (see Todo at beginning of script).
    """

    os.makedirs('Images/')    
    os.makedirs('PNG/')
    os.makedirs('Mask/')
    os.makedirs('Seg/')

    print('Folders created, move images into Images/ and update hash before rerunning command.')
    
    return


# makepng module

def makepng():
    """
    creates pngs from dicoms that can be used for segmentation. Saved in PNG/.
    """

    ds = dicom.dcmread(I[i], force=True)
    data = ds.pixel_array
    im = Image.fromarray(data/8)
    im = im.convert("L")
    im.save(P[i])
    
    return

# manual segmentation module

def manualseg():
    """
    uses cv2.bitwise_and to extract area of interest from png using mask. 
    Segmented pngs are saved in Seg/.
    """

    img = cv2.imread(P[i])
    mask = cv2.imread(M[i],0)
    seg = cv2.bitwise_and(img,img,mask=mask)
    cv2.imwrite(S[i], seg)
    
    return
    
    
# function call
# runs modules - note Todo hash.

ospath = os.getcwd()
os.chdir(sys.argv[1])    
    
folders()

'''
# runs loop based off csv

csv = 'Master.csv'
path = 'Images/'
png = 'PNG/'
mask = 'Mask/'
seg = 'Seg/

df = pd.read_csv(csv, header=0)
X = df.iloc[:,0]

for filename in X:
    I = path + filename + '.dcm'
    P = png + filename + '.png'
    M = mask + filename + '.png'
    S = seg + filename + '.png'
    
# runs functions
    
    for i in range(len(X)):
        makepng()
        manualseg()     
'''
