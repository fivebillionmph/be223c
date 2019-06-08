from util_cnn.util import read_data_unet
from model_cnn.model import createModel_Unet
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard


def train_seg(label_path, image_path, mask_path, tfbd_path, check_point_path, save_path):
    input_size = (256,256)
    img_dim = (256,256,1)
    batch_size = 32
    epoch_num = 50
    INIT_LR = 3e-4

    # for Unet segmentation training: 
    train_img, train_mask, train_label, val_img, val_mask, val_label = read_data_unet(label_path,image_path, mask_path, input_size, split_ratio=0.2, aug_rotate=6)
    
    model = createModel_Unet(*img_dim)  #256
	
    opt = Adam(lr=INIT_LR,decay=INIT_LR / epoch_num)
	# opt = Adam(lr=INIT_LR)

    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
	# model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=["categorical_accuracy"])

	#tensorboard
    tfbd = TensorBoard(log_dir=tfbd_path, histogram_freq=0,write_graph=True, write_images=True)

	# checkpoint
    ckptName = 'checkpoint{epoch:02d}-loss{loss:.2f}-val_loss{val_loss:.2f}-acc{acc:.2f}-val_acc{val_acc:.2f}.model'
    checkpoint = ModelCheckpoint(check_point_path+ckptName, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
	
    callbacks_list = [checkpoint,tfbd]

	# fit the data
    H = model.fit(train_img,train_mask, batch_size=batch_size, validation_data=(val_img,val_mask), epochs=epoch_num, verbose=1, callbacks=callbacks_list)

	#print(H.history)
	## Save the model and plot
    model.save(save_path+'lung_seg.model')
	# plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epoch_num
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Needle/not Needle")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(save_path+'plot.png')

    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.title('Loss Unet')
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(save_path+'plot_loss.png')

    plt.figure()
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title('Accuracy Unet')
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(save_path+'plot_accuracy.png')

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="train or infer")
    args = vars(ap.parse_args())

    label_path = '../Master3.csv'
    image_path = '../PNG-v2'
    mask_path = '../PNG-lung-seg'
    tfbd_path = '../tensorboard'
    check_point_path = '../checkpoint'
    save_path = '../save'

    if args['mode'] == 'train':
        train_seg(label_path, image_path, mask_path, tfbd_path, check_point_path, save_path)

    elif args['mode'] == 'infer':
        pass
		# infer_seg()

    else:
        print('Input correct parameters: train or infer')


