from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
#import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.datasets import cifar10
import sys
import csv
import shutil
import argparse
from keras.backend.tensorflow_backend import set_session
'''
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
'''

#trying to set gpu here but not working
'''
gpu = '0'

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
from keras import backend as K
K.tensorflow_backend.set_session(sess)
'''

##########
#package imports 
from models import siamese,siameseVAE
from utils import Siamese_Loader

from PIL import Image
import glob
import copy



def maxpooling(array,supersize,filename,infer):
    new_a=np.zeros_like(array,dtype=np.float32)
    for i in range(np.shape(array)[0]-supersize[0]):
        for j in range(np.shape(array)[1]-supersize[1]):
            patch=array[i:i+supersize[0],j:j+supersize[1]]
            new_a[i:i+supersize[0],j:j+supersize[1]]+=np.max(patch)/(supersize[0]*supersize[1])
    new_im = Image.fromarray(new_a.astype(np.uint8))
    '''
    if infer==True:
        new_im.save('test_data/blurred'+repr(supersize)+'/'+filename)
    else:
        new_im.save('testpooling'+repr(supersize)+'/'+filename)
    '''
    return new_a

def data_process(Filename, label, pooling_param, resize_param):
    image_list=[]
    labels=[]
    #print("In data process function")
    for filename in glob.glob(Filename): #assuming gif
        im=Image.open(filename)
        if len(np.shape(im))==3:
            im = im.convert('L')
        #im = im.resize((32, 32))
        #print("maxpooling callibration occupancy data ")
        array=np.array(im)
        #print("converted image shape ",array.shape)
        #array=maxpooling(array,pooling_param,filename,False)
        new_im = Image.fromarray(array)
        im = new_im.resize((resize_param, resize_param))

        im=np.array(im)
        im=np.reshape(im,(resize_param,resize_param,1))
        image_list.append(im)
        labels.append(label)
    return image_list, labels
def pickle_save(x_train,y_train,x_val,y_val,x_test,y_test):
    x_train_file = open('x_train_file.obj', 'wb')
    pickle.dump(x_train, x_train_file)
    x_train_file.close()

    y_train_file = open('y_train_file.obj', 'wb')
    pickle.dump(y_train, y_train_file)
    y_train_file.close()

    x_val_file = open('x_val_file.obj', 'wb')
    pickle.dump(x_val, x_val_file)
    x_val_file.close()

    y_val_file = open('y_val_file.obj', 'wb')
    pickle.dump(y_val, y_val_file)
    y_val_file.close()

    x_test_file = open('x_test_file.obj', 'wb')
    pickle.dump(x_test, x_test_file)
    x_test_file.close()

    y_test_file = open('y_test_file.obj', 'wb')
    pickle.dump(y_test, y_test_file)
    y_test_file.close()

def pickle_load():
    x_train_file = open('x_train_file.obj', 'rb')
    x_train = pickle.load(x_train_file)
    x_train_file.close()

    y_train_file = open('y_train_file.obj', 'rb')
    y_train = pickle.load(y_train_file)
    y_train_file.close()

    x_val_file = open('x_val_file.obj', 'rb')
    x_val = pickle.load(x_val_file)
    x_val_file.close()

    y_val_file = open('y_val_file.obj', 'rb')
    y_val = pickle.load(y_val_file)
    y_val_file.close()

    x_test_file = open('x_test_file.obj', 'rb')
    x_test = pickle.load(x_test_file)
    x_test_file.close()

    y_test_file = open('y_test_file.obj', 'rb')
    y_test = pickle.load(y_test_file)
    y_test_file.close()

    return x_train,y_train,x_val,y_val,x_test,y_test


def read_images(filename_occ, filename_unocc, pooling_param, resize_param,filename,ext):
    image_list_train = []
    image_list_val = []
    image_list_test = []

    y_train=[]
    y_val = []
    y_test = []

    if ext=="png":
        fext = '/*.png'
    if ext=="jpg":
        fext = '/*.jpg'

    '''
    try:
        x_train,y_train,x_val,y_val,x_test,y_test = pickle_load()
        return x_train,y_train,x_val,y_val,x_test,y_test
    except:
        print("Preprocessed files dont exist ")
    '''
    #base_location = 'Instances/H3/RS4/'
    #base_location = 'Instances/generalized/'
    base_location = 'Instances/'+filename+'/'

    x,y = data_process(base_location+'cal_data/'+filename_occ+fext, 1, pooling_param, resize_param) # 1 for occupied label
    image_list_train.extend(x)
    y_train.extend(y)

    x,y = data_process(base_location+'cal_data/'+filename_unocc+fext, 0, pooling_param, resize_param) # 1 for occupied label
    image_list_train.extend(x)
    y_train.extend(y)

    x,y = data_process(base_location+'sup_set/'+filename_occ+fext, 1, pooling_param, resize_param) # 1 for occupied label
    image_list_val.extend(x)
    y_val.extend(y)

    x,y = data_process(base_location+'sup_set/'+filename_unocc+fext, 0, pooling_param, resize_param) # 1 for occupied label
    image_list_val.extend(x)
    y_val.extend(y)

    x,y = data_process(base_location+'test_set/'+filename_occ+fext, 1, pooling_param, resize_param) # 1 for occupied label
    image_list_test.extend(x)
    y_test.extend(y)

    x,y = data_process(base_location+'test_set/'+filename_unocc+fext, 0, pooling_param, resize_param) # 1 for occupied label
    image_list_test.extend(x)
    y_test.extend(y)


    x_train= np.array(image_list_train)
    x_val= np.array(image_list_val)
    x_test= np.array(image_list_test)

    y_train= np.array(y_train)
    y_train= np.reshape(y_train,(len(y_train),1))

    y_val= np.array(y_val)
    y_val= np.reshape(y_val,(len(y_val),1))

    y_test= np.array(y_test)
    y_test= np.reshape(y_test,(len(y_test),1))

    #Save them to pickle for fast loading later on
    #pickle_save(x_train,y_train,x_val,y_val,x_test,y_test)

    return x_train,y_train,x_val,y_val,x_test,y_test

def reset(fname):
    '''
    Remove cal data folder from Instances
    copy images in test_set into cal_data
    '''
    old_cal_data_path = 'Instances/'+fname+'/cal_data'
    shutil.rmtree(old_cal_data_path)
    # shutil.copytree(from here, to there)
    shutil.copytree('Instances/'+fname+'/test_set','Instances/'+fname+'/cal_data')
def main(file,ext):

    pooling_param=[2,2]
    resize_param = 64
    filename = file #'H2'

    print("Got instance filename ",file)
    #sys.exit(0)
    reset(filename)
    
    def writelog(things):
        fname = 'Instances/'+filename+'/loss.csv'
        file1 = open(fname, 'a')
        writer = csv.writer(file1)
        fields1=things
        writer.writerow(fields1)
        file1.close()
    #WARNING ! before starting the training new again erase old data from cal_data folder and copy paste into it from test_set folder
    #Load the training data from cal_data folder
    #Load the validation data from sup_set folder
    #load the test data (actual provided labelled example which never change) from test_set folder
    x_train, y_train, x_val, y_val, x_test, y_test=read_images('occupied','unoccupied',pooling_param, resize_param, filename, ext)
    print("y_train shape ",y_train.shape)

    #create the siamese loader class
    loader=Siamese_Loader(x_train,x_val,x_test,y_train,y_val,y_test, resize_param,filename)

    sm=siamese(resize_param,1,False)
    

    #Initial warmstart training
    #If there are n+n annotated examples in test set, generate n^2 pairs and train with similarity/dissimilarity
    #If extremely similar images in the pair assign label of 1, 0 otherwise
    (inputs,targets)=loader.get_batch()
    for i in range(10):
        loss=sm.siamese_net.train_on_batch(inputs,targets)
        print("Trained network, got loss ",loss)

    '''
    say there are m annotates images in test set
    and n unlablled images in sup_set
    first generate m*n pairs (n pairs for each image in test set)
    then let the network predict the similarity/dissimilarity
    for each n pairs get an array of probabilities (we call them confidences)
    the mean of the array of probabilities is a theshold

    if for a pair the test image was occupied, and after pairing with an image from the support set, the output prob was p, with p>thresh
    then that specific index of the pair gets a +1 in confidence score in conf_array_occ

    if for a pair the test image was unoccupied, and after pairing with an image from the support set, the output prob was p, with p>thresh
    then that specific index of the pair gets a +1 in confidence score in conf_array_unocc

    (over all the m times n pairings) If on an average, that specific index in the pair has a confidence score> m/2 in conf_array_occ (agrees with at least half of the true labels)
    then the image in the support set corresponding to that specific pair gets saved to cal_data/occupied folder

    (over all the m times n pairings) If on an average, that specific index in the pair has a confidence score> m/2 in conf_array_unocc (agrees with at least half of the true labels)
    then the image in the support set corresponding to that specific pair gets saved to cal_data/unoccupied folder
    '''

    loader.test_oneshot_ability(sm.siamese_net,resize_param) #overwrites training images in cal_data folder| depends on images in sup_set and test_set

    tpc = 0.0
    #Now the clustering loop begins
    for i in range(10000):
        #M-step (in EM algorithm)
        #because of reassignment to cal_data folder the x_train and y_train changes
        x_train, y_train, x_val, y_val, x_test, y_test=read_images('occupied','unoccupied',pooling_param, resize_param, filename, ext) #x_train, y_train is now different
        print("y_train shape ",y_train.shape)

        (inputs,targets)=loader.get_batch()
        loss=sm.siamese_net.train_on_batch(inputs,targets)
        writelog([loss])
        sm.siamese_net.save_weights("Instances/"+filename+"/active_learning.h5")
        print("Trained network, got loss ",loss)

        #E-step in EM algorithm
        tpc = loader.test_oneshot_ability(sm.siamese_net,resize_param) #rewrites again

        if i%10==0:
            print("                                          This is 10th time !!!")
            #loader.reset()

def parse_args():
    parser = argparse.ArgumentParser("Few shot clustering for residential occupancy detection")
    # Environment
    parser.add_argument("--file", type=str, help="name of the zone where you want to do occupancy clustering")
    parser.add_argument("--fext", type=str, default = "png", help="image file extension- .png or .jpg?")

    args = parser.parse_args()
    return args.file, args.fext

if __name__ == '__main__':
    filen,fext = parse_args()
    main(filen,fext)

#python main.py --file "cent"