from keras.models import Sequential
from keras.layers import Input, Add, ZeroPadding2D, Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D, AveragePooling2D, Concatenate, Conv2DTranspose
from keras import backend as K
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.initializers import glorot_uniform
# from keras.applications.resnet50 import ResNet50
# from keras.applications.densenet import DenseNet121

def unet_convblock(input_tensor, filter_num, filter_size, p = 'same', batchnorm = 'True'):
    # filter_num : 64
    # filter_size : (3,3)
    # 

    # first conv layer
    X = Conv2D(filters=filter_num, kernel_size=filter_size, padding=p)(input_tensor)
    if batchnorm:
        X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)

    # second conv layer
    X = Conv2D(filters=filter_num, kernel_size=filter_size, padding=p)(X)
    if batchnorm:
        X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)

    return X   

def createModel_Unet(row,col,depth):

    inputShape = (row, col, depth)

	# if we are using "channels first", update the input shape
    if K.image_data_format() == "channels_first":
        inputShape = (depth, row, col)

    X_input = Input(inputShape)

    c1 = unet_convblock(X_input, 32, (3,3))                                         #256x256x32
    p1 = MaxPooling2D(pool_size=(2,2))(c1)                                          #128x128x32

    c2 = unet_convblock(p1, 64, (3,3))                                              #128x128x64
    p2 = MaxPooling2D(pool_size=(2,2))(c2)                                          #64x64x64

    c3 = unet_convblock(p2, 128, (3,3))                                             #64x64x128
    p3 = MaxPooling2D(pool_size=(2,2))(c3)                                          #32x32x128

    c4 = unet_convblock(p3, 256, (3,3))                                             #32x32x256
    p4 = MaxPooling2D(pool_size=(2,2))(c4)                                          #16x16x256

    c5 = unet_convblock(p4, 512, (3,3))                                             #16x16x512

    d6 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same')(c5)             #32x32x256
    d6 = Concatenate()([d6, c4])                                                    #32x32x512
    c6 = unet_convblock(d6, 256, (3,3))                                             #32x32x256

    d7 = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(c6)             #64x64x128
    d7 = Concatenate()([d7, c3])                                                    #64x64x256
    c7 = unet_convblock(d7, 128, (3,3))                                             #64x64x128

    d8 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(c7)              #128x128x64
    d8 = Concatenate()([d8, c2])                                                    #128x128x128
    c8 = unet_convblock(d8, 64, (3,3))                                              #128x128x64

    d9 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(c8)              #256x256x32
    d9 = Concatenate()([d9,c1])                                                     #256x256x64
    c9 = unet_convblock(d9, 32, (3,3))                                              #256x256x32

    output = Conv2D(1, (1,1), padding='same')(c9)
    output = Activation('sigmoid')(output)

    model = Model(inputs=X_input, outputs=output, name='Unet256')

    return model

def createModel_encoder(model_unet):

    encoder_unet = Model(inputs=model_unet.input, outputs=model_unet.get_layer('activation_10').output, name='encoder_unet'  )
    encoder = Sequential()
    encoder.add(encoder_unet)
    encoder.add(Flatten())
    encoder.add(Dense(256, activation='relu'))
    encoder.add(Dropout(0.1))
    encoder.add(Dense(32, activation='relu'))
    encoder.add(Dense(1, activation='sigmoid'))

    return encoder



def vgg16_classify(row, col, depth):
    
    input_shape = (row, col, depth)

    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in conv_base.layers[:-4]:
        layer.trainable = False
    
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    # model = createModel(256,256,3,2)
    # model = createModel_AlexNet(227,227,1,2)
    # model = createModel_ResNet18(229,229,1,2)
    # model = createModel_DensNet(224,224,1,2)
    model = createModel_Unet(128,128,1)
    encoder = createModel_encoder(model)
    # model = vgg16_classify(64,64,3)
    print(encoder.summary())
    # a = model.inputs
    # print(model.inputs)