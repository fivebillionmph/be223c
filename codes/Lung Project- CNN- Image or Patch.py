#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries

import os
# to run on GPU, comment the following tow lines 
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing import image
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.applications import VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import pylab
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from util3 import read_data # with image
from util5 import read_data_random_view # with patch
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import random


# In[6]:


# inputs

# if you want to train on images, choose RC = 1
# if you want to train on patches, choose RC = 2
RC = 2 # running code

# input shape for the VGG16 (it sould be the same as patch size)

input_shape=(64, 64, 3)

# train

csvPath = 'OLD2/Master2M.csv'
imagePath = '/home/maalidoost/OLD2/Seg2M'
patchPath = '/home/maalidoost/OLD2/Seg2Mask'

inputSize = (224, 224)
patchSize = (64, 64)
split_ratio = 0.2
batch_size = 32
learning_rate = 1e-6
epoch_size = 2
nodes = 512 # number of nodes for the fc layer
drop_out = 0.5 # after the fc layer

# test

test_dir = '/home/maalidoost/Seg2'
csvTest = 'Master2.csv'


# In[9]:


# preparing the data to train

random.seed(7)

if RC == 1:
    train_img, train_label, val_img, val_label = read_data(csvPath,
            imagePath, inputSize, split_ratio, split_by_id = True, normalize = True, crop_image = True)
    train_img = np.stack( (train_img[:,:,:,0],)*3, axis=-1 )
    val_img = np.stack( (val_img[:,:,:,0],)*3, axis=-1 )
    print("")
    print("using images")
    print("")
    print("train_images:", train_img.shape)
    print("train_labels:", train_label.shape)
    print("validation_images:", val_img.shape)
    print("validation_label:", val_label.shape)
    print("")
elif  RC == 2:
    train_img, train_label, val_img, val_label =     read_data_random_view(csvPath, imagePath, patchPath, inputSize, patchSize, split_ratio, kfold=1, outchannels=3)
    print("")
    print("using patches")
    print("")
    print('train_patches:', train_img.shape)
    print('train_labels:', train_label.shape)
    print('validation_patches:', val_img.shape) 
    print('validation_labels:', val_label.shape)
    print("")

print("getting the model")
print("")
# using the pretrained model for training

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
#conv_base.summary()

for layer in conv_base.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers

#for layer in conv_base.layers:
#    print(layer, layer.trainable)

# making the FC layers of the model

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(nodes, activation='relu'))
model.add(layers.Dropout(drop_out))
model.add(layers.Dense(1, activation='sigmoid'))
#model.summary()

# compiling the model

model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=learning_rate), metrics=['acc'])

# data augmentation

train_datagen = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

#val_datagen = ImageDataGenerator(rescale=1./255)

# training the model

H = model.fit_generator(
        train_datagen.flow(train_img, train_label, batch_size = batch_size),
        steps_per_epoch=len(train_img) // batch_size,
        epochs = epoch_size,
        validation_data = (val_img,val_label),
        validation_steps = len(val_img) // batch_size)

# plotting the training loss and accuracy

print("")
plt.style.use("ggplot")
plt.figure()
N = epoch_size
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

# saving the model

model.save_weights('model_weights_lung_pro.h5')
model.save('model_lung_pro.h5')


# In[4]:


# testing the model

# loading the model

model = load_model('model_lung_pro_2.h5')
#model.summary()

# getting X and y for testing

file_list2 = os.listdir(test_dir)
test_imgs = [test_dir + "/" + "{}".format(i) for i in file_list2]
#print("No. of test images = ", len(test_imgs))

X_test = []
IMG_SIZE = 224
for image in test_imgs:
    X_test.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC))

X_test = np.array(X_test)
X_test = X_test/255.0
print("shape of X_test:", X_test.shape)

df = pandas.read_csv(csvTest)
#print('shape of the dataframe:', df.shape)
#print(df.head(2))

na = df.loc[:,'File']
la = df.loc[:,'Progression']
na = np.array(na)
la = np.array(la)
I = np.argsort(na)
na = na[I]
la = la[I]
y_test = la

sns.set(rc={'figure.figsize':(5,4)})
sns.countplot (y_test)
plt.title("Labels")
print ("shape of y_test:", y_test.shape)

# model prediction

preds_test = model.predict(X_test, verbose=1)
preds_test_t = (preds_test > 0.5).astype(np.uint8)
predictions_test = preds_test_t [:, 0]
print("")

print("Predicted labels:", predictions_test)
print("")
print("True labels:", y_test)
print("")

com = np.isclose(predictions_test, y_test.T)
print (com)

#true_prediction_number = 1 * (com == 'True')
#print(true_prediction_number)

cm = confusion_matrix(y_test, predictions_test)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

