## utility function for cnn
import pandas as pd
import numpy as np
import cv2
from os.path import join
from sklearn.model_selection import train_test_split
from operator import itemgetter
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import KFold
from scipy import ndimage
import matplotlib.pyplot as plt
import random

random.seed(7)

def normalize(img):
    # Normalize the images in the range of 0 to 1 (converted into float64)

    return (img-np.min(img))/(np.max(img)-np.min(img))
    # return img / np.max(img)

def normalize_u8(img):
    # Normalize the images in the range of 0 to 255 (converted into uint8)

    return np.uint8( 255*normalize(img) )    

def bbox(img):
    # find the bounding box of the lung in ct slices
    
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def read_label(csv_file):
    # read image name and store in list

    label_csv = pd.DataFrame( pd.read_csv(csv_file) )

    file_name_list = list(label_csv['File'])
    prog_list = list(label_csv['Progression'])
    pid_file_dict = dict()
    for i,name in enumerate(file_name_list):
        pid = name[1:3]
        if not pid in pid_file_dict:
            pid_file_dict[pid] = [i]
        else:
            pid_file_dict[pid].append(i)
    pid_list = list( pid_file_dict.keys() )

    # print(pid_list)
    return file_name_list,prog_list,pid_list,pid_file_dict

# read all images and masks according to csvfile 

# OUTPUT: 
#       image_list: (list) list of numpy arrays of image with specified size
#       mask_list: (list) list of numpy arrays of mask with specified size  
def read_image(file_name_list, image_src_path, mask_src_path, size):
 
    image_list = list()
    mask_list = list()

    for f in file_name_list:
        img = cv2.imread( join(image_src_path,(f+'.png')) , cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread( join(mask_src_path,(f+'.png')) , cv2.IMREAD_GRAYSCALE)
        
        img_rz = cv2.resize(img, (size[1],size[0]))
        mask_rz = cv2.resize(mask, (size[1],size[0]))
        
        img_rz = normalize(img_rz)
        mask_rz = np.float64( mask_rz>0.5 ) 
        
        image_list.append(img_rz)
        mask_list.append(mask_rz)

    return image_list, mask_list


## Simply replicate labels by n times
def augment_label(label, n):
    return [label]*n



## rotate the images n times and wrap them in a list, rotation will crop corners
## INPUT: 
##      image ( row x col)
## OUTPUT: 
##      list (n) with each element as (row x col)
def augment_rotate(img, n):

    
    aug_patch = list()

    angle_step = round(180/n) 
    for ang in range(0,180,angle_step):
        img_rot = rotate(img, ang)
        aug_patch.append( img_rot )
    
    return aug_patch

## INPUT:
##      mask: numpy array 2d 
## OUTPUT:
##      coordianates: tuple (row,col)
def find_lesion_coordiates(mask):


    row,col = ndimage.measurements.center_of_mass(mask) 

    return (int(row), int(col))  

## extract patch from image according to coordinates
## INPUT
##      img: numpy array (M,N) 
##      coor: tuple (row,col) 
##      patch_size: int (has to be smaller than img)
## OUTPUT
##      patch: numpy array (patch_size,patch_size)
def extract_patch(img,coor,patch_size):

    rmax, cmax = img.shape

    row, col = coor
    r1 = row - round( 0.5*(patch_size[0]) )
    r2 = r1 + patch_size[0]
    c1 = col - round( 0.5*(patch_size[1]) )
    c2 = c1 + patch_size[1]

    if r1 < 0:
        r1 = 0
        r2 = patch_size[0]
    elif r2 > rmax:
        r1 = rmax - patch_size[0]
        r2 = rmax
    
    if c1 < 0:
        c1 = 0
        c2 = patch_size[1]
    elif c2 > cmax:
        c1 = cmax - patch_size[1]
        c2 = cmax
    
    return img[r1:r2,c1:c2]

## extract patch from image according to coordinates (rewrite of extract_patch())
## INPUT
##      img: numpy array (M,N) 
##      coor: tuple (row,col) 
##      patch_size: int (has to be smaller than img)
## OUTPUT
##      patch: numpy array (patch_size,patch_size)
def generate_patch_from_img(img,coor,patch_size):
    
    return extract_patch(img, coor, patch_size)



## Augment the patches with random view sampling
##      Shifting, Rotating, Scaling
## INPUT
##      img: numpy array (M,N) 
##      coor: tuple (row,col) 
##      patch_size: int (has to be smaller than img)
## OUTPUT
##      patch_list: list (including N patches)
##      N: int (augmentation times)
def generate_patch_from_img_random_views(img,coor,patch_size):
    
    
    shift_step = round(0.2 * patch_size[0])
    shifting_r =  [-shift_step, 0, shift_step]
    shifting_c = [-shift_step, 0, shift_step]
    rotating = [0, 36, 72, 108, 144]
    scaling = [0.7, 1.0, 1.3]
    Nt = len(shifting_r) * len(shifting_c)
    Nr = len(rotating)
    Ns = len(scaling)
    N = Nt*Nr*Ns


    r0,c0 = coor

    patch_list = list()

    for dr in shifting_r:
        r = r0 + dr
        for dc in shifting_c:
            c = c0 + dc
            for ro in rotating:
                for sc in scaling:
                    img_transform = rotate(img, ro, center=(c,r), scale=sc)
                    patch_list.append( extract_patch(img_transform,(r,c),patch_size) )

    
    return patch_list, N


def read_data_unet(label_path,image_folder_path,mask_folder_path,input_size,split_ratio=0.2,aug_rotate=6):
## read data for proposed unet-based method
##      No option for K-Fold; No option for RBG/gra output (one channel in default); 
##      Split on patient ID; Augmentation: rotation(crop corners)

## OUTPUT
##      numpy array: train_img (N x row x col x 1), train_mask (N x row x col x 1), train_label (N,)
##      numpy array: val_img (N x row x col x 1), val_mask (N x row x col x 1), val_label (N,)



    ## read all the data
    file_name_list,prog_list,pid_list,pid_file_dict = read_label(label_path)
    #print('Labels loaded: {} positive,{} negatve.'.format( sum( np.array(prog_list)==1 ), len(prog_list)-sum( np.array(prog_list)==1 #)))
    
    image_list, mask_list = read_image(file_name_list, image_folder_path, mask_folder_path, input_size)
    #print('Images loaded: {}; Masks loaded: {}'.format( len(image_list), len(mask_list) ))

    ## define data capacities
    train_img = list()
    train_mask = list()
    train_label = list()
    val_img = list()
    val_mask = list()
    val_label = list()

    ## split on patient id and query image according to pid_file_dict
    ## augmentation with rotation
    train_id, val_id = train_test_split( pid_list, test_size=split_ratio, random_state=42)
    for pid in train_id:
        for ind in pid_file_dict[pid]:
            train_img += augment_rotate(image_list[ind], aug_rotate)
            train_mask += augment_rotate(mask_list[ind], aug_rotate)
            train_label += augment_label(prog_list[ind], aug_rotate)  

    for pid in val_id:
        for ind in pid_file_dict[pid]:
            val_img += augment_rotate(image_list[ind], aug_rotate)
            val_mask += augment_rotate(mask_list[ind], aug_rotate)
            val_label += augment_label(prog_list[ind], aug_rotate)  


    print('Split dataset according to Patients. Training : {} patients and {} images; Validation : {} patients and {} images'.format( 
        len(train_id),len(train_img),len(val_id),len(val_img) ))

    train_img = np.expand_dims(np.array(train_img), axis=-1)
    train_mask = np.expand_dims(np.array(train_mask), axis=-1)
    train_label = np.array(train_label)

    val_img = np.expand_dims(np.array(val_img), axis=-1)
    val_mask = np.expand_dims(np.array(val_mask), axis=-1)
    val_label = np.array(val_label)

    return train_img, train_mask, train_label, val_img, val_mask, val_label    



def read_data_dual_input(label_path,image_folder_path,mask_folder_path,input_size1,input_size2=(64,64),split_ratio=0.2,aug_rotate=6,kfold=1,outchannels=3):
## read data for proposed dual input VGG 16 method
##      kfold: 1,2,3,4,5...
##      Option: K-Fold(default: 1, no cross validation) ; outchannels (RBG/gray output) (default: 3 (RGB)); 
##      Split on patient ID; Augmentation: rotation(crop corners)

## OUTPUT
##      if not KFold:
##      numpy array: train_img1 (N x row1 x col1 x outchannels), train_img2 (N x row2 x col2 x outchannels), train_label (N,)
##      numpy array: val_img1 (N x row1 x col1 x outchannels), val_img2 (N x row2 x col2 x outchannels), val_label (N,)
##      if KFold:
##      numpy array: train_img1 (K x N x row1 x col1 x outchannels), train_img2 (K x N x row2 x col2 x outchannels), train_label (K, N)
##      numpy array: val_img1 (K x N x row1 x col1 x outchannels), val_img2 (K x N x row2 x col2 x outchannels), val_label (K, N) 
# 
#     
    ## read all the data
    file_name_list,prog_list,pid_list,pid_file_dict = read_label(label_path)
    #print('Labels loaded: {} positive,{} negatve.'.format( sum( np.array(prog_list)==1 ), len(prog_list)-sum( np.array(prog_list)==1 #)))
    
    image_list, mask_list = read_image(file_name_list, image_folder_path, mask_folder_path, input_size1)
    #print('Images loaded: {}; Masks loaded: {}'.format( len(image_list), len(mask_list) ))



    if kfold > 1:
        train_img1_kfold = list()
        train_img2_kfold = list()
        train_label_kfold = list()
        val_img1_kfold = list()
        val_img2_kfold = list()
        val_label_kfold = list()

        kf = KFold(n_splits=kfold)
        pid_list = np.array(pid_list) # for convenient indexing 
        for train_id_index, val_id_index in kf.split(pid_list):
            train_id, val_id = pid_list[train_id_index], pid_list[val_id_index]
            
            train_img1 = list()
            train_img2 = list()
            train_label = list()
            val_img1 = list()
            val_img2 = list()
            val_label = list()

            for pid in train_id:
                for ind in pid_file_dict[pid]:
                    img_batch = augment_rotate(image_list[ind],aug_rotate)
                    mask_batch = augment_rotate(mask_list[ind],aug_rotate)

                    train_img1 += img_batch
                    for i in range(len(img_batch)):
                        train_img2.append( generate_patch_from_img(img_batch[i], find_lesion_coordiates(mask_batch[i]), patch_size=input_size2))
                    train_label += augment_label(prog_list[ind], aug_rotate)  

            for pid in val_id:
                for ind in pid_file_dict[pid]:
                    img_batch = augment_rotate(image_list[ind],aug_rotate)
                    mask_batch = augment_rotate(mask_list[ind],aug_rotate)

                    val_img1 += img_batch
                    for i in range(len(img_batch)):
                        val_img2.append( generate_patch_from_img(img_batch[i], find_lesion_coordiates(mask_batch[i]), patch_size=input_size2))
                    val_label += augment_label(prog_list[ind], aug_rotate)  


            train_img1_kfold.append( np.stack( [np.array(train_img1)] * outchannels, axis=-1 ) )
            train_img2_kfold.append( np.stack( [np.array(train_img2)] * outchannels, axis=-1 ) )
            train_label_kfold.append( np.array(train_label) )

            val_img1_kfold.append( np.stack( [np.array(val_img1)] * outchannels, axis=-1 ) )
            val_img2_kfold.append( np.stack( [np.array(val_img2)] * outchannels, axis=-1 ) )
            val_label_kfold.append( np.array(val_label))

        train_img1_kfold_np = np.array(train_img1_kfold)
        train_img2_kfold_np = np.array(train_img2_kfold)
        train_label_kfold_np = np.array(train_label_kfold)

        val_img1_kfold_np = np.array(val_img1_kfold)
        val_img2_kfold_np = np.array(val_img2_kfold)
        val_label_kfold_np = np.array(val_label_kfold)
        
        return train_img1_kfold_np,train_img2_kfold_np,train_label_kfold_np,val_img1_kfold_np,val_img2_kfold_np,val_label_kfold_np

    else:  # no kfold, kfold == 1

        ## define data capacities           
        train_img1 = list()
        train_img2 = list()
        train_label = list()
        val_img1 = list()
        val_img2 = list()
        val_label = list()

        ## split on patient id and query image according to pid_file_dict
        ## augmentation with rotation
        train_id, val_id = train_test_split( pid_list, test_size=split_ratio, random_state=42)
        for pid in train_id:
            for ind in pid_file_dict[pid]:
                img_batch = augment_rotate(image_list[ind],aug_rotate)
                mask_batch = augment_rotate(mask_list[ind],aug_rotate)

                train_img1 += img_batch
                for i in range(len(img_batch)):
                    train_img2.append( generate_patch_from_img(img_batch[i], find_lesion_coordiates(mask_batch[i]), patch_size=input_size2))
                train_label += augment_label(prog_list[ind], aug_rotate)  


        for pid in val_id:
            for ind in pid_file_dict[pid]:
                img_batch = augment_rotate(image_list[ind],aug_rotate)
                mask_batch = augment_rotate(mask_list[ind],aug_rotate)

                val_img1 += img_batch
                for i in range(len(img_batch)):
                    val_img2.append( generate_patch_from_img(img_batch[i], find_lesion_coordiates(mask_batch[i]), patch_size=input_size2))
                val_label += augment_label(prog_list[ind], aug_rotate)  
        
        train_img1_np =  np.stack( [np.array(train_img1)] * outchannels, axis=-1 ) 
        train_img2_np =  np.stack( [np.array(train_img2)] * outchannels, axis=-1 ) 
        train_label_np = np.array(train_label) 

        val_img1_np =  np.stack( [np.array(val_img1)] * outchannels, axis=-1 ) 
        val_img2_np =  np.stack( [np.array(val_img2)] * outchannels, axis=-1 ) 
        val_label_np = np.array(val_label)

        return train_img1_np,train_img2_np,train_label_np,val_img1_np,val_img2_np,val_label_np
        




def read_data_random_view(label_path,image_folder_path,mask_folder_path,input_size1,input_size2=(64,64),split_ratio=0.2,kfold=1,outchannels=3):
## read data for proposed patch input VGG 16 with random view sampling method
##      kfold: 1,2,3,4,5...
##      Option: K-Fold(default: 1, no cross validation) ; outchannels (RBG/gray output) (default: 3 (RGB)); 
##      Split on patient ID; Augmentation: random view sampling

## OUTPUT
##      if not KFold:
##      numpy array: train_img (N x row x col x outchannels), train_label (N,)
##      numpy array: val_img (N x row x col x outchannels), val_label (N,)
##      if KFold:
##      numpy array: train_img (K x N x row x col x outchannels),train_label (K, N)
##      numpy array: val_img (K x N x row x col x outchannels), val_label (K, N) 
# 
#    
    ## read all the data
    file_name_list,prog_list,pid_list,pid_file_dict = read_label(label_path)
    #print('Labels loaded: {} positive,{} negatve.'.format( sum( np.array(prog_list)==1 ), len(prog_list)-sum( np.array(prog_list)==1 #)))
    
    image_list, mask_list = read_image(file_name_list, image_folder_path, mask_folder_path, input_size1)
    #print('Images loaded: {}; Masks loaded: {}'.format( len(image_list), len(mask_list) ))



    if kfold > 1:
        train_img_kfold = list()
        train_label_kfold = list()
        val_img_kfold = list()
        val_label_kfold = list()

        kf = KFold(n_splits=kfold)
        pid_list = np.array(pid_list) # for convenient indexing 
        for train_id_index, val_id_index in kf.split(pid_list):
            train_id, val_id = pid_list[train_id_index], pid_list[val_id_index]
            
            train_img = list()
            train_label = list()
            val_img = list()
            val_label = list()

            for pid in train_id:
                for ind in pid_file_dict[pid]:
                    img = image_list[ind]
                    coor = find_lesion_coordiates(mask_list[ind])
                    patch,N = generate_patch_from_img_random_views(img,coor,input_size2)
                    train_img += patch 
                    train_label += augment_label(prog_list[ind], N)

            for pid in val_id:
                for ind in pid_file_dict[pid]:
                    img = image_list[ind]
                    coor = find_lesion_coordiates(mask_list[ind])
                    patch,N = generate_patch_from_img_random_views(img,coor,input_size2)
                    val_img += patch 
                    val_label += augment_label(prog_list[ind], N)


            train_img_kfold.append( np.stack( [np.array(train_img)] * outchannels, axis=-1 ) )
            train_label_kfold.append( np.array(train_label) )

            val_img_kfold.append( np.stack( [np.array(val_img)] * outchannels, axis=-1 ) )
            val_label_kfold.append( np.array(val_label))

        train_img_kfold_np = np.array(train_img_kfold)
        train_label_kfold_np = np.array(train_label_kfold)

        val_img_kfold_np = np.array(val_img_kfold)
        val_label_kfold_np = np.array(val_label_kfold)
        
        return train_img_kfold_np, train_label_kfold_np, val_img_kfold_np, val_label_kfold_np

    else:  # no kfold, kfold == 1

        ## define data capacities           
        train_img = list()
        train_label = list()
        val_img = list()
        val_label = list()

        ## split on patient id and query image according to pid_file_dict
        ## augmentation with rotation
        train_id, val_id = train_test_split( pid_list, test_size=split_ratio, random_state=42)
        for pid in train_id:
            for ind in pid_file_dict[pid]:
                img = image_list[ind]
                coor = find_lesion_coordiates(mask_list[ind])
                patch,N = generate_patch_from_img_random_views(img,coor,input_size2)
                train_img += patch 
                train_label += augment_label(prog_list[ind], N)

        for pid in val_id:
            for ind in pid_file_dict[pid]:
                img = image_list[ind]
                coor = find_lesion_coordiates(mask_list[ind])
                patch,N = generate_patch_from_img_random_views(img,coor,input_size2)
                val_img += patch 
                val_label += augment_label(prog_list[ind], N)
        
        train_img_np =  np.stack( [np.array(train_img)] * outchannels, axis=-1 ) 
        train_label_np = np.array(train_label) 

        val_img_np =  np.stack( [np.array(val_img)] * outchannels, axis=-1 ) 
        val_label_np = np.array(val_label)

        return train_img_np, train_label_np, val_img_np, val_label_np


















## export the following functions


def read_image_test(image_path, input_size):
## read a single image
##     image_path: string,the path to the single mask
##     input_size: tuple, (row,col)

##     return: numpy array(float64), the image

## example

##     image_path = '../img.png'
##     input_size = (224,224)
##     img = read_image_test(image_path, input_size)  



    img = cv2.imread(image_path , cv2.IMREAD_GRAYSCALE)
    img_rz = cv2.resize(img, (input_size[1],input_size[0]))

    return normalize(img_rz)



def read_mask_test(mask_path, input_size):
## read a single mask
##     image_path: string,the path to the single mask
##     input_size: tuple, (row,col)

##     return: numpy array(float64), the binary mask


## example

##     mask_path = '../mask.png'
##     input_size = (224,224)
##     mask = read_mask_test(mask_path, input_size)  
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_rz = cv2.resize(mask, (size[1],size[0]))
    
    return np.float64( mask_rz>0.5 )     




def read_patch_test(img,mask,patch_size=(64,64)):
## extract a patch from a image based on lesion mask
##     img: numpy array, the lung image
##     mask: numpy array, the lesion mask
##     patch_size: tuple, (row,col)

##     return: numpy array(float64), the patch


## example

##     image_path = '../img.png'
##     input_size = (224,224)
##     img = read_image_test(image_path, input_size)  

##     mask_path = '../mask.png'
##     input_size = (224,224)
##     mask = read_mask_test(mask_path, input_size)  

##     patch = read_patch_test(img,mask,patch_size=(64,64))

    return extract_patch(img, find_lesion_coordiates(mask), patch_size)





def read_patch_rvs_test(img,mask,patch_size=(64,64)):
## extract a list of patches from a image based on lesion mask
##     img: numpy array, the lung image
##     mask: numpy array, the lesion mask
##     patch_size: tuple, (row,col)

##     return: a list of numpy array(float64), the patches

## example

##     image_path = '../img.png'
##     input_size = (224,224)
##     img = read_image_test(image_path, input_size)  

##     mask_path = '../mask.png'
##     input_size = (224,224)
##     mask = read_mask_test(mask_path, input_size)  

##     patches = read_patch_rvs_test(img,mask,patch_size=(64,64)) # return a list of patches

    p,n = generate_patch_from_img_random_views(img, find_lesion_coordiates(mask), patch_size)
    return p


def gray2RGB(im):
## convert a 2-D numpy array to 3-channel numpy array
## im (numpy array): (row,col)

## example:
#   im = cv2.imread('../im.png','cv2.IMREAD_GRAYSCALE')
#   im2 = gray2RGB(im)   #im2 (row,col,3)

    return np.stack( [im] * 3, axis=-1 ) 





#######################################################################################
def read_data(label_path,image_folder_path,input_size,split_ratio,aug_rotate=6,split_by_id=True,normalize=True,crop_image=False):
    file_name_list,prog_list,pid_list,pid_file_dict = read_label(label_path)
    #print('Labels loaded: {} positive,{} negatve.'.format( sum( np.array(prog_list)==1 ), len(prog_list)-sum( np.array(prog_list)==1 #)))
    
    image_list = read_image(file_name_list, image_folder_path, input_size, norm=normalize, crop=crop_image)  
    angleInc = round(180/aug_rotate)

    train_img = list()
    train_label = list()
    val_img = list()
    val_label = list()
    if split_by_id:
        train_id, val_id = train_test_split( pid_list, test_size=split_ratio, random_state=42)
        for pid in train_id:
            for ind in pid_file_dict[pid]:
                im = image_list[ind]
                for ang in range(0,180,angleInc):
                    im_rot = rotate(im, ang)
                    train_img.append( img_to_array(im_rot) )
                    train_label.append( prog_list[ind] )

        for pid in val_id:
            for ind in pid_file_dict[pid]:
                im = image_list[ind]
                for ang in range(0,180,angleInc):
                    im_rot = rotate(im, ang)
                    val_img.append( img_to_array(im_rot) )
                    val_label.append( prog_list[ind] )
        print('Split dataset according to Patients. Training : {} patients and {} images; Validation : {} patients and {} images'.format( 
            len(train_id),len(train_img),len(val_id),len(val_img) ))
    else:
        train_img_temp, val_img_temp, train_label_temp, val_label_temp = train_test_split(image_list, 
        prog_list, test_size=split_ratio, random_state=42)

        for i in range(len(train_img_temp)):
            for ang in range(0,180,angleInc):
                im_rot = rotate(train_img_temp[i], ang)
                train_img.append( img_to_array(im_rot) )
                train_label.append( train_label_temp[i] )      

        for i in range(len(val_img_temp)):
            for ang in range(0,180,angleInc):
                im_rot = rotate(val_img_temp[i], ang)
                val_img.append( img_to_array(im_rot) )
                val_label.append( val_label_temp[i] )    


        # train_img = np.expand_dims(train_img, axis=-1)
        # val_img = np.expand_dims(val_img, axis=-1)
        print('Split dataset according to images. Training : {} images; Validation: {} images.'.format(
            len(train_img),len(val_img) ))

    train_img_np = np.array(train_img)
    train_label_np = np.array(train_label)
    val_img_np = np.array(val_img)
    val_label_np = np.array(val_label)

    return train_img_np,train_label_np,val_img_np,val_label_np

def read_data_kfold(label_path,image_folder_path,input_size,aug_rotate=6,kfold=5,normalize=True,crop_image=False):
    ## it will split the dataset only according to the patient id in k-fold manner
    ## for each image, it will do manual augmentation by rotation
    ## crop is optional and with risk that it may lose some important corners
    ## most of the parameters are defined in the same way as read_data()
    
    
    file_name_list,prog_list,pid_list,pid_file_dict = read_label(label_path)
    #print('Labels loaded: {} positive,{} negatve.'.format( sum( np.array(prog_list)==1 ), len(prog_list)-sum( np.array(prog_list)==1 #)))
    
    image_list = read_image(file_name_list, image_folder_path, input_size, norm=normalize, crop=crop_image)  
    angleInc = round(180/aug_rotate)

    train_img_kfold = list()
    train_label_kfold = list()
    val_img_kfold = list()
    val_label_kfold = list()

    kf = KFold(n_splits=kfold)
    pid_list = np.array(pid_list)
    for train_id_index, val_id_index in kf.split(pid_list):
        train_id, val_id = pid_list[train_id_index], pid_list[val_id_index]
        train_img = list()
        train_label = list()
        val_img = list()
        val_label = list()
        for pid in train_id:
            for ind in pid_file_dict[pid]:
                im = image_list[ind]
                for ang in range(0,180,angleInc):
                    im_rot = rotate(im, ang)
                    train_img.append( img_to_array(im_rot) )
                    train_label.append( prog_list[ind] )
        for pid in val_id:
            for ind in pid_file_dict[pid]:
                im = image_list[ind]
                for ang in range(0,180,angleInc):
                    im_rot = rotate(im, ang)
                    val_img.append( img_to_array(im_rot) )
                    val_label.append( prog_list[ind] )
        train_img_kfold.append( np.array(train_img) )
        train_label_kfold.append( np.array(train_label) )
        val_img_kfold.append( np.array(val_img) )
        val_label_kfold.append( np.array(val_label) )

    train_img_kfold_np = np.array(train_img_kfold)
    train_label_kfold_np = np.array(train_label_kfold)
    val_img_kfold_np = np.array(val_img_kfold)
    val_label_kfold_np = np.array(val_label_kfold)

    # print('Split dataset according to Patients. Training : {} patients and {} images; Validation : {} patients and {} images'.format( 
    #     len(train_id),len(train_img),len(val_id),len(val_img) ))
    return train_img_kfold_np, train_label_kfold_np, val_img_kfold_np, val_label_kfold_np
#######################################################################################



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))


if __name__ == "__main__":
    csvPath = '../Master3.csv'
    imagePath = '../PNG-v2'
    maskPath = '../PNG-lung-seg'
    inputSize = (256,256)
    # patchSize = (64,64)


    # for Unet segmentation training: 
    a1,b1,c1,d1,e1,f1 = read_data_unet(csvPath,imagePath, maskPath, inputSize, split_ratio=0, aug_rotate=1)
    print('Unet training data preparation:')
    print('Training:', a1.shape, b1.shape, c1.shape)
    print('Valiadation:', d1.shape, e1.shape, f1.shape)

    # # for dual inputs (original image and small patch of leision) training:
    # a2,b2,c2,d2,e2,f2 = read_data_dual_input(csvPath,imagePath,imagePath,inputSize,patchSize,split_ratio=0.2,aug_rotate=6,kfold=1,outchannels=3)
    # print('Dual inputs training data preparation:')
    # print('Training:', a2.shape, b2.shape, c2.shape)
    # print('Valiadation:', d2.shape, e2.shape, f2.shape)

    # # for single input (small patch with random view sampling) training: 
    # a3,b3,c3,d3 = read_data_random_view(csvPath,imagePath,imagePath,inputSize,patchSize,split_ratio=0.2,kfold=1,outchannels=3)
    # print('Random view sampling patch input training data preparation:')
    # print('Training:', a3.shape, b3.shape)
    # print('Valiadation:', c3.shape, d3.shape) 
   

    ## utility function for usage in testing and web backend

    ##find_lesion_coordiates(mask)
    ##generate_patch_from_img(img,coor,patch_size)
    ##generate_patch_from_img_random_views(img,coor,patch_size)
