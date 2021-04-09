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
num_actual_labels = 5 #the 'n' representative images value of n
# required for validating (for this case need two folders in testing '0' 
# containing v representative images and '1' containing v representative images)
num_validation_labels = 28 #the 'v' representative images value of v
num_sampled_ul = 50 #A parameter of the algorithm - number of unlabeled images paired with labeled images each time
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

def get_loss(output, target):
    #BCE loss
    l = nn.BCELoss()
    return l(output,target)


def input_pairs(): #used for sampling pairs in the main E-M algorithm
    names = image_names()
    label_keys = list(names.keys())

    label_pairs = {}
    

    n_ul = random.sample(names[label_keys[-1]], num_sampled_ul) # n_unlabeled, the last key stores the unlabelled images names
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
                
                '''
                else: #randomly swap the image ordering to the siamese network as well
                pair_l.append(img_ul) #input to left half of the siamese network
                pair_ul.append(img_l) #input to right half of the siamese network
                '''


        #pair_l, pair_ul = tensorize(pair_l), tensorize(pair_ul)

        label_pairs[n] = {'labeled':pair_l,'unlabeled':pair_ul}

    return label_pairs, unlabeled_sampled_names


def input_pairs_labeled(): #used for light pretraining

    names = image_names()
    
    label_keys = list(names.keys())

    label_pairs = {}


    targets = {i:[] for i in labels}

    

    for n1 in label_keys[:-1]:
        pair_l1 = []
        pair_l2 = []

        n_l1 = names[n1]
        for n2 in label_keys[:-1]:
            
            n_l2 = names[n2]

            for img_name1 in n_l1:
                img_l1 = read_image(img_name1,(32,32),True,True) #get a normalized numpy array of the image
                for img_name2 in n_l2:
                    img_l2 = read_image(img_name2,(32,32),True,True)

                    pair_l1.append(img_l1) #input to left half of the siamese network
                    pair_l2.append(img_l2) #input to right half of the siamese network

                    if n1==n2:
                        targets[n1].append(1)
                    else:
                        targets[n1].append(0)

        label_pairs[n1] = {'labeled':pair_l1,'unlabeled':pair_l2}

    return label_pairs,targets

def input_pairs_validation(): #used for validation
    names_labeled = image_names(folder = 'labeled')
    names_validation = image_names(folder = 'validation')

    label_keys = list(names_labeled.keys())
    label_pairs = {}

    targets = {i:[] for i in labels}
    for n1 in label_keys[:-1]:
        pair_l1 = []
        pair_l2 = []

        n_l1 = names_labeled[n1]
        for n2 in label_keys[:-1]:
            
            n_l2 = names_validation[n2]

            for img_name1 in n_l1:
                img_l1 = read_image(img_name1,(32,32),True,True) #get a normalized numpy array of the image
                for img_name2 in n_l2:
                    img_l2 = read_image(img_name2,(32,32),True,True)

                    pair_l1.append(img_l1) #input to left half of the siamese network
                    pair_l2.append(img_l2) #input to right half of the siamese network

                    if n1==n2:
                        targets[n1].append(1)
                    else:
                        targets[n1].append(0)

        label_pairs[n1] = {'labeled':pair_l1,'unlabeled':pair_l2}

    return label_pairs,targets




def E_step(model,label_pairs, validating):

    def percent_threshold_values(array,percent=50):
        p = percent/100.0
        #given 2d array, looks at each row of the array
        #for each row, sort the row and pick the (percent)th value in the ascending order of the row
        #finally return a column vector for the 2d matrix
        col = []
        for i in range(array.shape[0]):
            row = array[i,:]
            row_sort = np.sort(row)
            cut = int(p*array.shape[1])
            v = row_sort[cut]
            col.append(v)
        return np.array(col)




    print("\nE step")
    nsul = num_sampled_ul
    nal = num_actual_labels

    if validating:
        nsul = num_validation_labels*2
        #nal = num_validation_labels
        nal = num_actual_labels

    label_score = {i:np.zeros(len(labels),) for i in range(nsul)}

    for l in labels:
        img_l = tensorize(label_pairs[l]['labeled']).to(device)
        img_ul = tensorize(label_pairs[l]['unlabeled']).to(device)

        s_m = model(img_l, img_ul).detach().cpu().numpy()
        similarity_estimates = s_m



        #now similarity_estimates[l] is a vector of shape -> ((number of images present in each folder in labels/) * num_sampled_ul, 1)
        #now need to reshape the similarity_estimates list into class specific tables (see the arxiv paper)

        #similarity_table = similarity_estimates.reshape(-1,nsul)
        similarity_table = similarity_estimates.reshape(nal,nsul)
        #print("similarity table shape ",similarity_table.shape)
        a = similarity_table
        #see this technique 
        #(get the median in each row of the table and threshold each row values to either 0 or 1 based or lesser or greater than row median)
        #https://stackoverflow.com/questions/53106728/how-to-threshold-based-on-the-average-value-of-a-row
        
        a[a >= np.broadcast_to(percent_threshold_values(a),(a.shape[1],a.shape[0])).T] = 1.0 # '>=' is very important/ wont work if only '>' is used
        a = np.where(a==1.0,a,0.0)

        #now get the sum of the thresholded values for each column (converts table to a list)
        b = a.sum(axis=0).tolist()
        for i in range(len(b)):
            label_score[i][l] = b[i] #each image index i in the unlabeled set, for each unique label l, 
            #b[i] is the number of times the unlabeled image from index 'i' had a high similarity (as predicted by the model)
            #with the labeled image that has the label l

    return label_score #label_score will be later used to come up with fictitious labels for unlabeled images based on the majority voting

def M_step(model,label_score,validating, label_pairs = []):
    nsul = num_sampled_ul
    nal = num_actual_labels
    foldr = 'labeled'
    if validating:
        nsul = num_validation_labels*2
        #nal = num_validation_labels
        nal = num_actual_labels

    most_likely_labels = np.zeros(nsul,)
    #print(label_score)
    for l in label_score.keys():
        most_likely_labels[l] = np.argmax(label_score[l])

    targets = {i:[] for i in labels}

    for l in labels: 
        #n_l = names[l]
        #n_ul = names[label_keys[-1]]

        for i in range(nal):
            for j in range(nsul):

                #optionally visualize what the model is predicting
                if label_pairs!=[]:
                    img_l = label_pairs[l]['labeled'][i*nsul+j]
                    img_ul = label_pairs[l]['unlabeled'][i*nsul+j]
                    pred_label = most_likely_labels[j]
                    actual_label = l
                    #cv2.imwrite('data/model_pred/'+repr(int(pred_label))+'/labeled_'+repr(i)+'_'+repr(j)+'.png', img_l*255.0)
                    cv2.imwrite('data/model_pred/'+repr(int(pred_label))+'/unlabeled_'+repr(j)+'.png', img_ul*255.0)

                if most_likely_labels[j]==l:
                    targets[l].append(1) #similar
                else:
                    targets[l].append(0) #dissimilar
    #print("got targets ",targets)
    return targets


def train_targets(model,label_pairs, targets, warmstart):
    #num_sampled_l = int(len(targets[0])/num_sampled_ul)
    nsl = int(len(targets[0])/num_sampled_ul)
    nsul = num_sampled_ul
    
    if warmstart:
        nsl = num_actual_labels
        nsul = num_actual_labels
    

    #batches are sampled in a interleaved manner
    '''
    This is the structure of label_pairs
    (KEY)                   'labeled'           'unlabeled'
    unique_label_id (0) -> labeled_image_1 | unlabeled_image_1
                           labeled_image_1 | unlabeled_image_2
                           labeled_image_1 | unlabeled_image_3
                           ...
                           ...
                           labeled_image_1 | unlabeled_image_50 (50 = num_sampled_ul= numble of unlabeled images sampled everytime)

                           labeled_image_2 | unlabeled_image_1
                           labeled_image_2 | unlabeled_image_2
                           ...
                           labeled_image_2 | unlabeled_image_50
                           ...
                           ...
                           ...
                           labeled_image_8 | unlabeled_image_50 (say there are total 8 images with a definite label= unique_label_id = 0)

    unique_label_id (1) -> labeled_image_1 | unlabeled_image_1
                           labeled_image_1 | unlabeled_image_2
                           labeled_image_1 | unlabeled_image_3
                           ...
                           ...
                           labeled_image_1 | unlabeled_image_50 (50 = num_sampled_ul)

                           labeled_image_2 | unlabeled_image_1
                           labeled_image_2 | unlabeled_image_2
                           ...
                           labeled_image_2 | unlabeled_image_50
                           ...
                           ...
                           ...
                           labeled_image_8 | unlabeled_image_50 (say there are total 8 images with a definite label= unique_label_id = 1)

    
    for the first batch, 
    we will take 
    unique_label_id (0)|labeled_image_1 | unlabeled_image_1
    unique_label_id (1)|labeled_image_1 | unlabeled_image_1

    unique_label_id (0)|labeled_image_2 | unlabeled_image_1
    unique_label_id (1)|labeled_image_2 | unlabeled_image_1
    ...
    ...
    unique_label_id (0)|labeled_image_8 | unlabeled_image_1
    unique_label_id (1)|labeled_image_8 | unlabeled_image_1

    for the next batch we will take

    unique_label_id (0)|labeled_image_1 | unlabeled_image_2
    unique_label_id (1)|labeled_image_1 | unlabeled_image_2

    unique_label_id (0)|labeled_image_2 | unlabeled_image_2
    unique_label_id (1)|labeled_image_2 | unlabeled_image_2
    ...
    ...
    unique_label_id (0)|labeled_image_8 | unlabeled_image_2
    unique_label_id (1)|labeled_image_8 | unlabeled_image_2

    and so on thus we take number of batches = number of unlabeled images sampled everytime

    targets is basically 0 or 1 depending on whether the pair is similar or not. It is stored as a flat list 
    correspondence is derived based on the structure of label_pairs explained above

    '''
    for b in range(nsul): #train across a number of extracted mini batches
        img_l = []
        img_ul = []
        batch_targets = []

        for n in range(nsl):
            for l in labels:
                idx = b+(n*nsul)
                #print("got index ",idx)
                img_l.append(label_pairs[l]['labeled'][idx])
                img_ul.append(label_pairs[l]['unlabeled'][idx])
                batch_targets.append(targets[l][idx])

        img_l = tensorize(img_l).to(device)
        img_ul = tensorize(img_ul).to(device)
        batch_targets = torch.tensor(batch_targets,dtype = torch.float).to(device)


        #print("got batch targets ",batch_targets)
        model_pred = model(img_l,img_ul)
        optimizer.zero_grad()
        train_loss = get_loss(model_pred, batch_targets)
        #print("model pred ",model_pred)
        #print("batch targets ",batch_targets)
        train_loss.backward()
        optimizer.step()
        print("\rM step: batch number : {} , train_loss : {} ".format(b, train_loss),end = '')





def train():
    '''
    #initial testing
    names = get_names('data/labeled')
    print("got file names ",names)
    a = read_image(names[0],(256,128), True, True)
    test_image(a)
    '''


    
    num_warm_start = 10
    label_pairs, targets = input_pairs_labeled()
    print("Warm start ")
    for w in range(num_warm_start): 
        print("epoch ",w)
        train_targets(model,label_pairs,targets, True)
        torch.save(model.state_dict(), 'model.pth')
    
    
    

    print("Starting the clustering algorithm ...")
    num_clustering_steps = 500000
    
    for c in range(num_clustering_steps):
        label_pairs, _ = input_pairs()
        label_scores = E_step(model,label_pairs,False)

        targets = M_step(model,label_scores,False,label_pairs = label_pairs)
        train_targets(model,label_pairs,targets, False)
        torch.save(model.state_dict(), 'model.pth')

        if c%10==0:
            print("\nTesting on validation images ")
            validate()


def validate():
    label_pairs, targets = input_pairs_validation()
    label_scores = E_step(model,label_pairs,True)
    targets_estimate = M_step(model,label_scores,True)
    #print("target ",targets[0])
    #print("targets_estimate ",targets_estimate[0])
    label_wise_match = {}
    for l in targets.keys():
        num_wrong = np.sum(np.abs(np.array(targets[l])-np.array(targets_estimate[l])))
        f = (len(targets_estimate[l])- num_wrong)/len(targets_estimate[l])
        label_wise_match[l] = f

    print("Fraction of correct matches by label ")
    for l in label_wise_match.keys():
        print("label ",l," -> ",label_wise_match[l])
    #sys.exit(0)
        



if __name__ == '__main__':
    train()