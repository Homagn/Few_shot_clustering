from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Reshape, UpSampling2D
from keras.layers import add
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.utils import np_utils
from keras.models import load_model
import numpy.random as rng
import numpy as np
import os
#import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.datasets import cifar10
import sys

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)



class siamese(object):
    def __init__(self,img_size,img_channels,load_weights):
        self.imsize=img_size
        self.img_channels=img_channels
        input_shape = (self.imsize, self.imsize, self.img_channels)
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        #build convnet to use in each siamese 'leg'
        
        self.model = Sequential()
        #self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1),activation='relu',input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(32, (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        #self.model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=input_shape))
        #self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #self.model.add(Conv2D(64, (3, 3), activation='relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        #self.model.add(Dropout(0.2))
        #self.model.add(Dense(50, activation='relu'))
        
        #encode each of the two inputs into a vector with the convnet
        encoded_l = self.model(left_input)
        encoded_r = self.model(right_input)
        
        #new keras code
        #merge two encoded inputs with the l1 distance between them
        minus_encoded_r = Lambda(lambda x: -x)(encoded_r)
        subtracted = add([encoded_l,minus_encoded_r])
        both = Lambda(lambda x: K.abs(x))(subtracted)
        
        #old keras code
        #L1_distance = lambda x: K.abs(x[0]-x[1])
        #both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])

        prediction = Dense(50,activation='sigmoid')(both)
        prediction = Dense(50,activation='sigmoid')(prediction)
        #self.model.add(Dropout(0.2))
        prediction = Dense(20,activation='sigmoid')(prediction)
        prediction = Dense(1,activation='sigmoid')(prediction)

        self.siamese_net = Model(input=[left_input,right_input],output=prediction)
        #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

        optimizer = Adam(0.00006)
        #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
        self.siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

        self.siamese_net.count_params()
        #siamese_net.summary()
        print("Built and compiled siamese net")
        if(load_weights==True):
            #self.siamese_net.load_weights('weights/model_weight[10, 10]')
            self.siamese_net.load_weights('weights/model_weight[1, 1].h5')
            print("sucessfully loaded weights")

class siameseVAE(object):
    def __init__(self,img_size,img_channels,intermediate_dim,latent_dim,batch_size):
        self.imsize=img_size
        self.img_channels=img_channels
        input_shape = (self.imsize, self.imsize, self.img_channels)
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        #build convnet to use in each siamese 'leg'
        self.model = Sequential()

        self.model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(8, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(8, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten()) #Dimension here should be 128
        #self.model.add(Dense(500, activation='relu'))
        #self.model.add(Dense(100, activation='relu'))
        #self.model.add(Dense(10, activation='relu'))
        #encode each of the two inputs into a vector with the convnet
        encoded_l = self.model(left_input)
        encoded_r = self.model(right_input)
        #merge two encoded inputs with the l1 distance between them
        L1_distance = lambda x: K.abs(x[0]-x[1])
        #both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
        both = merge([encoded_l,encoded_r], mode = 'concat', output_shape=lambda x: x[0])
        h=Dense(intermediate_dim,activation='relu',bias_initializer=b_init)(both)
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            #epsilon = K.random_normal(shape=(batch_size, latent_dim),mean=0., std=0.01)
            epsilon = K.random_normal(shape=(batch_size, latent_dim))
            return z_mean + K.exp(z_log_sigma) * epsilon

        def vae_loss(x, x_decoded_mean):
            margin=2.0
            KL_loss_img= K.sum((x * x_decoded_mean) +((1 - x) * K.maximum(margin - x_decoded_mean, 0)))
            #xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            #return xent_loss + kl_loss
            return KL_loss_img + kl_loss

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma]) #VAE trick
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(128, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)
        encoded = Reshape((4, 4, 8))(x_decoded_mean)

        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


        #prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
        self.siamese_net = Model(input=[left_input,right_input],output=decoded)
        #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

        optimizer = Adam(0.00006)
        #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
        #self.siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
        self.siamese_net.compile(loss=vae_loss,optimizer=optimizer)

        self.siamese_net.count_params()
        #siamese_net.summary()
        print("Built and compiled siamese VAE net")
    