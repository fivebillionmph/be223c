'''
Training, testing of encoder-based lesion classificaiton model
Author: Zhaoqiang Wang (github: aaronzq)
'''

from model_cnn.model import vgg16_classify, createModel_encoder
from util_cnn.util import read_data_random_view, read_data_unet, normalize
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from os.path import join
import random
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,confusion_matrix


def train_classify(label_path, image_path, mask_path, save_path, input_size=(64,64)  ):
    
    epoch_num = 100
    learning_rate = 1e-6
    batch_size = 32

    train_img, train_label, val_img, val_label = read_data_random_view(label_path,image_path,mask_path, 
        input_size1=(224,224), input_size2=input_size, split_ratio=0.2, kfold=1, outchannels=3)

    model = vgg16_classify(*input_size,3)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate,decay=learning_rate / epoch_num), metrics=['acc'])

    # train_datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    train_datagen = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    H = model.fit_generator(
        train_datagen.flow(train_img, train_label, batch_size = batch_size),
        steps_per_epoch = len(train_img)//batch_size,
        epochs = epoch_num, validation_data = (val_img,val_label), validation_steps = len(val_img) // batch_size)

    
    
    model.save(join(save_path,'lesion_classify.model'))
	# plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epoch_num
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on lesion")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(join(save_path+'plot.png'))
    
    return H



def train_encoder_classify(unet_model_path, label_path, image_path, mask_path, tfbd_path, check_point_path, save_path, img_dim = (128,128,1) ):
    
    batch_size = 32
    epoch_num = 30
    INIT_LR = 3e-5

    # for Unet segmentation training: 
    train_img, train_mask, train_label, val_img, val_mask, val_label = read_data_unet(label_path, image_path, mask_path, (img_dim[0],img_dim[1]), split_ratio=0.2, aug_rotate=6)
  
    model = createModel_encoder(load_model(unet_model_path))

    opt = Adam(lr=INIT_LR,decay=INIT_LR / epoch_num)
	# opt = Adam(lr=INIT_LR)

    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
	# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=["categorical_accuracy"])

	#tensorboard
    tfbd = TensorBoard(log_dir=tfbd_path, histogram_freq=0,write_graph=True, write_images=True)

	# checkpoint
    ckptName = 'checkpoint{epoch:02d}-loss{loss:.2f}-val_loss{val_loss:.2f}-acc{acc:.2f}-val_acc{val_acc:.2f}.model'
    checkpoint = ModelCheckpoint(join(check_point_path, ckptName), monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
	
    callbacks_list = [checkpoint,tfbd]

	# fit the data
    H = model.fit(train_img, train_label, batch_size=batch_size, validation_data=(val_img,val_label), epochs=epoch_num, verbose=1, callbacks=callbacks_list)

	#print(H.history)
	## Save the model and plot
    model.save(join(save_path,'lesion_classification.model'))
	# plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epoch_num
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on lesion classification")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(join(save_path+'plot.png'))

    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title('Loss Unet')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(join(save_path+'plot_loss.png'))

    plt.figure()
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title('Accuracy Unet')
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(join(save_path+'plot_accuracy.png'))    

def test_encoder_classify(label_path, image_path, mask_path, load_path, save_path, img_dim = (128,128,1) ):

    test_image,test_mask,test_label,[],[],[] = read_data_unet(label_path,image_path, mask_path, (img_dim[0],img_dim[1]), split_ratio=0, aug_rotate=6)

    model = load_model(join(load_path,'lesion_classification.model'))
    test_results = model.predict(test_image)

    AUC = roc_auc_score(test_label,test_results)
    fpr, tpr, _ = roc_curve(test_label, test_results)
    plt.figure()
    plt.plot(fpr,tpr,label="encoder, auc="+str(AUC))
    plt.legend(loc=4)
    
    accuracy = accuracy_score(test_label,test_results>0.5)
    cm = confusion_matrix(test_label,test_results>0.5)    

    print('Accuracy : {}; AUC: {}'.format(accuracy, AUC)  )
    plt.show()










###########################################################################################
########################### export the following functions ################################
###########################################################################################




def infer_encoder_classify(img, model_path):
    """
    predict a probability of progression from a (lung-segmented) lung image
    
    Args: img: 2d numpy array, lung-segmented lung images on grayscale
           model_path: str of path to the model
    
    return: progression (float):  probability of progression
    
    ----------------------------- example ---------------------------------------
        test_image_path = '../Dataset/Test-Seg-Man/'
        img = cv2.imread( join(test_image_path,('A40PR2002'+'.png')) , cv2.IMREAD_GRAYSCALE)
        progression = infer_encoder_classify(img, join(save_path, 'lesion_classification.model'))
    -----------------------------------------------------------------------------
    """


    model = load_model(model_path)
    input_size = ( int(model.inputs[0].shape[1]), int(model.inputs[0].shape[2]) )

    img_rz = normalize(cv2.resize(img, (input_size[1],input_size[0])))
    img_rz = np.expand_dims(img_rz, axis=-1)
    img_rz = np.expand_dims(img_rz, axis=0)

    return float(model.predict(img_rz)[0])

###########################################################################################
###########################################################################################
###########################################################################################








if __name__ == "__main__":

    unet_model_path = '../save/lung_seg.model'

    train_label_path = '../Dataset/Train.csv'
    train_image_path = '../Dataset/Train-Seg-Man/'  #'../Dataset/Train-PNG/'
    train_mask_path = '../Dataset/Train-Patch/'
    test_label_path = '../Dataset/Test.csv'
    test_image_path = '../Dataset/Test-Seg-Man/'
    test_mask_path = '../Dataset/Test-Patch/'

    tfbd_path = '../tensorboard/'
    check_point_path = '../checkpoint/'
    save_path = '../save/'
    test_save_path = '../test/'

    # train_encoder_classify(unet_model_path, train_label_path, train_image_path, train_mask_path, tfbd_path, check_point_path, save_path, img_dim = (128,128,1) )   
    # test_encoder_classify(test_label_path, test_image_path, test_mask_path, save_path, test_save_path, img_dim = (128,128,1) )
    

    
    img = cv2.imread( join(test_image_path,('A40PR2002'+'.png')) , cv2.IMREAD_GRAYSCALE)
    progression = infer_encoder_classify(img, join(save_path, 'lesion_classification.model'))


    pass



    # train_classify( label_path, image_path, mask_path, save_path, input_size=(64,64))
