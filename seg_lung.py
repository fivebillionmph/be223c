from util_cnn.util import read_data_unet,normalize
from model_cnn.model import createModel_Unet
import argparse
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard


def train_seg(label_path, image_path, mask_path, tfbd_path, check_point_path, save_path, img_dim = (128,128,1)):
    
    batch_size = 32
    epoch_num = 50
    INIT_LR = 3e-4

    # for Unet segmentation training: 
    train_img, train_mask, train_label, val_img, val_mask, val_label = read_data_unet(label_path,image_path, mask_path, (img_dim[0],img_dim[1]), split_ratio=0.2, aug_rotate=6)
    
    model = createModel_Unet(*img_dim)  #256
	
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
    H = model.fit(train_img, train_mask, batch_size=batch_size, validation_data=(val_img,val_mask), epochs=epoch_num, verbose=1, callbacks=callbacks_list)

	#print(H.history)
	## Save the model and plot
    model.save(join(save_path,'lung_seg.model'))
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

def test_seg(label_path, image_path, mask_path, load_path, save_path, img_dim = (128,128,1)):

    test_image,test_mask,test_label,[],[],[] = read_data_unet(label_path,image_path, mask_path, (img_dim[0],img_dim[1]), split_ratio=0, aug_rotate=1)

    model = load_model(join(load_path,'lung_seg.model'))
    test_results = model.predict(test_image)

    jaccard = list()
    dice = list()

    for i in range(len(test_results)):
        test_im = np.float64(test_results[i,:,:,0]>.5)
        gt_im = test_mask[i,:,:,0]
        intersection = np.sum(np.logical_and(test_im, gt_im))
        union = np.sum(np.logical_or(test_im, gt_im))
        jaccard.append( intersection / union )
        dice.append( 2*intersection/( np.sum(test_im) + np.sum(gt_im) ) )

        cv2.imwrite(join(save_path+'/net/'+str(i)+'.tif'),np.uint8(255*test_im))
        cv2.imwrite(join(save_path+'/gt/'+str(i)+'.tif'),np.uint8(255*gt_im))
    
    print('Average jaccard: {}; Average dice: {}'.format(np.mean(jaccard), np.mean(dice))  )



## predict a mask from a lung image
# img (2d numpy array): lung images on grayscale
# model_path (str): path to the model

# return: mask (2d numpy array): mask with the same size of input image 
def infer_seg(img, model_path):
    ## ----------------------------- example ---------------------------------------
    # image_path = '../Dataset/Test-PNG/'
    # model_path = '../save/'

    # img = cv2.imread( join(image_path,('A40PR2002'+'.png')) , cv2.IMREAD_GRAYSCALE)
    # mask = infer_seg(img, join(model_path, 'lung_seg.model'))
    ## -----------------------------------------------------------------------------

    output_size = img.shape

    model = load_model(model_path)
    input_size = ( int(model.inputs[0].shape[1]), int(model.inputs[0].shape[2]) )

    img_rz = normalize(cv2.resize(img, (input_size[1],input_size[0])))
    img_rz = np.expand_dims(img_rz, axis=-1)
    img_rz = np.expand_dims(img_rz, axis=0)
    mask_output = cv2.resize( model.predict(img_rz)[0,:,:,0], (output_size[1], output_size[0]) )
    mask_output = np.float64(mask_output>.5)
    return mask_output


if __name__ == "__main__":

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="train or infer")
    args = vars(ap.parse_args())

    train_label_path = '../Dataset/Train.csv'
    train_image_path = '../Dataset/Train-PNG/'
    train_mask_path = '../Dataset/Train-Seg-Mask/'
    test_label_path = '../Dataset/Test.csv'
    test_image_path = '../Dataset/Test-PNG/'
    test_mask_path = '../Dataset/Test-Seg-Mask/'


    tfbd_path = '../tensorboard/'
    check_point_path = '../checkpoint/'
    save_path = '../save/'
    test_save_path = '../test/'

    # img = cv2.imread( join(image_path,('A40PR2002'+'.png')) , cv2.IMREAD_GRAYSCALE)
    # mask = infer_seg(img, join(save_path, 'lung_seg.model'))

    pass

    if args['mode'] == 'train':
        train_seg(train_label_path, train_image_path, train_mask_path, tfbd_path, check_point_path, save_path)

    elif args['mode'] == 'test':
        test_seg(test_label_path, test_image_path, test_mask_path, save_path, test_save_path)  

    else:
        print('Input correct parameters: train or infer')


