import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import glob
import random

from models import Siamese
import sys


labels = [0,1] #number of distinct labels in the labeled dataset
# required for training (for this case need two folders in labeled '0' 
# containing n representative images and '1' containing n representative images)
num_actual_labels = 20 #the 'n' representative images value of n
# required for validating (for this case need two folders in testing '0' 
# containing v representative images and '1' containing v representative images)
num_validation_labels = 28 #the 'v' representative images value of v
num_sampled_ul = 100 #A parameter of the algorithm - number of unlabeled images paired with labeled images each time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Siamese((32,32),1) #image size (32,32), 1 channel
try:
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    print("successfully loaded weights ")
except:
    print("Failed to load weights ")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=6e-5)




def get_names(folder):
    return glob.glob(folder+'/*.png')

def read_image(fname, size, normalize, mono):
    if mono:
        img = cv2.imread(fname,0)
    else:
        img = cv2.imread(fname)

    img = cv2.resize(img,size)
    if normalize:
        return np.array(img)/255.0
    return np.array(img)

def test_image(array):
    print("array looks like ",array)
    print("image dimensions : width = ",array.shape[1]," height = ",array.shape[0])
    cv2.imshow('loaded image',array)
    cv2.waitKey(0)


def image_names(folder = 'labeled'):
    
    labeled_images = {}

    for i in labels:
        try:
            labeled_images[i] = get_names('data/'+folder+'/'+repr(i))
        except:
            print("The labeled folder ",i," does not exist in the data/labeled ")

    #final key stores the unlabeled images
    labeled_images[labels[-1]+1] = get_names('data/unlabeled')
    return labeled_images

def tensorize(array_list):
    ar = np.array(np.stack(array_list,axis=0), dtype = np.float32)
    bsize = ar.shape[0]
    h = ar.shape[1]
    w = ar.shape[2]
    try:
        c = ar.shape[3]
    except:
        c = 1
    return torch.tensor(ar).view((bsize, 1, w, h))


def E_step_infer(model,label_pairs):
    print("\n Inference E step")
    nsul = 1 #inference on 1 image
    nal = num_actual_labels


    label_score = []

    for l in labels:
        img_l = tensorize(label_pairs[l]['labeled']).to(device)
        img_ul = tensorize(label_pairs[l]['unlabeled']).to(device)

        s_m = model(img_l, img_ul).detach().cpu().numpy()
        similarity_estimates = s_m

        similarity_table = similarity_estimates.reshape(nal,nsul)
        a = similarity_table

        label_confidence = np.sum(a) #get a single value denoting the confidence of the model in that single image being the specific label
        label_score.append(label_confidence)

    return label_score #label_score will be later used to come up with fictitious labels for unlabeled images based on the majority voting





def inference_pair(inf_image, labeled_folder = 'labeled'): #used for sampling pairs in the main E-M algorithm
    labeled_images = {}

    for i in labels:
        try:
            labeled_images[i] = get_names('data/'+labeled_folder+'/'+repr(i))
        except:
            print("The labeled folder ",i," does not exist in the data/labeled ")

    #final key stores the unlabeled images
    labeled_images[labels[-1]+1] = [inf_image] #list containing the single file name of the image to infer
    names = labeled_images


    label_keys = list(names.keys())
    label_pairs = {}
    

    n_ul = [inf_image] #random.sample(names[label_keys[-1]], num_sampled_ul) # n_unlabeled, the last key stores the unlabelled images names
    unlabeled_sampled_names = n_ul

    for n in label_keys[:-1]:
        n_l = names[n] #n_labeled, stores the particular image names for that label
        

        pair_l = []
        pair_ul = []

        for img_name in n_l:
            img_l = read_image(img_name,(32,32),True,True) #get a normalized numpy array of the image
            for img_name_ul in n_ul:
                img_ul = read_image(img_name_ul,(32,32),True,True)

                #if random.random()>0.5:
                pair_l.append(img_l) #input to left half of the siamese network
                pair_ul.append(img_ul) #input to right half of the siamese network


        label_pairs[n] = {'labeled':pair_l,'unlabeled':pair_ul}

    return label_pairs

def classify_single_image(fname): #fname is the filename of the image we want to infer
    label_pairs = inference_pair(fname)
    label_scores = E_step_infer(model,label_pairs)
    print("got label scores ",label_scores)


def classify_localize(fname):
    names = [fname+'img'+str(k)+'.png' for k in range(5063)]
    frame = 0
    for f in names:
        img = read_image(f,(256,32),True,True)
        print("img shape ",img.shape)
        for i in range(8):
            start_point = (i*32,0)
            end_point = ((i+1)*32,32)
            color = (0, 0, 0)
            thickness = 5

            fname = 'test.png'
            frag = img[:,i*32:(i+1)*32]
            cv2.imwrite('test.png',frag*255.0)
            label_pairs = inference_pair(fname)
            label_scores = E_step_infer(model,label_pairs)
            print("got label scores ",label_scores)
            if label_scores[0]<=label_scores[1] and i!=1 and i!=2 and i!=3 and i!=4:
                img = cv2.rectangle(img, start_point, end_point, color, thickness)
            #cv2.imshow("fragment",frag)
            #cv2.waitKey(0)

        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow("sample",img)
        cv2.waitKey(1)
        cv2.imwrite('/data/extracted_classif/'+str(frame)+'.png',img*255.0)
        frame+=1




if __name__ == '__main__':
    #classify_single_image('test.png')
    classify_localize('extracted/')
