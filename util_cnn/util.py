## utility function for cnn
import pandas as pd
import numpy as np
import cv2
from os.path import join
from sklearn.model_selection import train_test_split
from operator import itemgetter
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt



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

def read_image(file_name_list, image_src_path, size, norm=True, crop=False):
    image_list = list()

    for f in file_name_list:
        img = cv2.imread( join(image_src_path,(f+'.png')) , cv2.IMREAD_GRAYSCALE)
        
        if crop:
            rmin, rmax, cmin, cmax = bbox(img)
            if rmax-rmin > cmax-cmin:
                cmax = cmin + rmax-rmin
            else:
                rmax = rmin + cmax-cmin
            img_rz = cv2.resize(img[rmin:rmax, cmin:cmax], (size[1],size[0]))
        else:
            img_rz = cv2.resize(img, (size[1],size[0]))

        if norm:
            img_rz = normalize(img_rz)
        
        image_list.append(img_rz)
    
    return image_list

def read_data(label_path,image_folder_path,input_size,split_ratio,split_by_id=True,normalize=True,crop_image=False):
    file_name_list,prog_list,pid_list,pid_file_dict = read_label(label_path)
    print('Labels loaded: {} positive,{} negatve.'.format( sum( np.array(prog_list)==1 ), len(prog_list)-sum( np.array(prog_list)==1 )))
    
    image_list = read_image(file_name_list, image_folder_path, input_size, norm=normalize, crop=crop_image)  

    train_img = list()
    train_label = list()
    val_img = list()
    val_label = list()
    if split_by_id:
        train_id, val_id = train_test_split( pid_list, test_size=split_ratio, random_state=42)
        for pid in train_id:
            for ind in pid_file_dict[pid]:
                train_img.append( img_to_array(image_list[ind]) )
                train_label.append( prog_list[ind] )
        for pid in val_id:
            for ind in pid_file_dict[pid]:
                val_img.append( img_to_array(image_list[ind]) )
                val_label.append( prog_list[ind] )
        print('Split dataset according to Patients. Training : {} patients and {} images; Validation : {} patients and {} images'.format( 
            len(train_id),len(train_img),len(val_id),len(val_img) ))
    else:
        train_img, val_img, train_label, val_label = train_test_split(image_list, 
        prog_list, test_size=split_ratio, random_state=42)
        train_img = np.expand_dims(train_img, axis=-1)
        val_img = np.expand_dims(val_img, axis=-1)
        print('Split dataset according to images. Training : {} images; Validation: {} images.'.format(
            len(train_img),len(val_img) ))

    train_img_np = np.array(train_img)
    train_label_np = np.array(train_label)
    val_img_np = np.array(val_img)
    val_label_np = np.array(val_label)

    return train_img_np,train_label_np,val_img_np,val_label_np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.sqrt(np.dot(vec1, vec1)) * np.sqrt(np.dot(vec2, vec2)))


if __name__ == "__main__":
    csvPath = '../Master.csv'
    imagePath = '../Seg'
    inputSize = (227,227)


    # fileNameList,progList,pidList,pidFileDict = read_label(csvPath)   
    # imageList = read_image(fileNameList, imagePath, inputSize, norm=True, crop=True) 

    # read_data(csvPath,imagePath,inputSize,0.3,split_by_id=True,normalize=True,crop_image=False)

    train_img,train_label,val_img,val_label = read_data(csvPath,
        imagePath,inputSize,0.3,split_by_id=True,normalize=True,crop_image=False)
        

    # print(type(train_img[5])) 