from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.datasets import cifar10
import sys
import random

import copy

from PIL import Image

import os
import glob

import time as t
from datetime import datetime as dt
import csv





class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,Xtrain,Xval,Xtest,Ytrain,Yval,Ytest,resize_param,filename): #Adapted slightly to load directly from cifar10
        numclasses=len(np.unique(Ytrain))

        self.resize_param = resize_param

        self.Xtrain=np.zeros((numclasses,len(Xtrain),resize_param,resize_param,1))
        print("Got number of distinct classes as ",numclasses)
        print("Got new shape of Xtrain as ",self.Xtrain.shape)
        self.valid_tuple=[]
        for i in range(len(Ytrain)): #Warning !! not an efficient implementation, dictionary is recommended 
            self.Xtrain[Ytrain[i],i,:,:,:]=Xtrain[i] #(label, index in dataset and image information )
            self.valid_tuple.append([Ytrain[i],i]) #(label, index in dataset)
        #print("Got new Xtrain as ",self.Xtrain)
        self.Xval = Xval
        self.Yval=Yval

        self.Xtest = Xtest
        self.Ytest=Ytest
        #self.Xtrain = Xtrain
        self.n_classes=numclasses
        self.n_examples,self.w,self.h,_ = Xtrain.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape
        self.numgood = 0
        self.numbad = 0
        self.filename = filename
        self.numways = len(Yval)

        #self.occ_preds = np.zeros((self.numways,)) #store the accumulated prediction
        #self.unocc_preds = np.zeros((self.numways,)) #store the accumulated prediction
        self.selected_occ = np.zeros((self.numways,)) #store the accumulated prediction
        self.selected_unocc = np.zeros((self.numways,)) #store the accumulated prediction

        self.expected_correct_hits = []

        self.numruns = 0

        print("Got self.numways ",self.numways)
        #sys.exit(0)

    def writeNthlog(self,things):
        if self.numruns%10==0:
            fname = 'Instances/'+self.filename+'/progressNth.csv'
            file1 = open(fname, 'a')
            writer = csv.writer(file1)
            fields1=things
            writer.writerow(fields1)
            file1.close()
            self.reset()
    def writelog(self, things):
        fname = 'Instances/'+self.filename+'/progress.csv'
        file1 = open(fname, 'a')
        writer = csv.writer(file1)
        fields1=things
        writer.writerow(fields1)
        file1.close()
    def reset(self):
        #self.occ_preds = np.zeros((self.numways,)) #store the accumulated prediction
        #self.unocc_preds = np.zeros((self.numways,)) #store the accumulated prediction
        self.selected_occ = np.zeros((self.numways,)) #store the accumulated prediction
        self.selected_unocc = np.zeros((self.numways,)) #store the accumulated prediction
        self.expected_correct_hits = self.expected_correct_hits[-100:]
    def get_batch(self):
        """Create batch of n pairs, half same class, half different class"""
        n = min(50,len(self.valid_tuple))
        categories = rng.choice(self.n_classes,size=(n,),replace=True)
        valid_choice=rng.choice(len(self.valid_tuple),size=(n,),replace=False) #chose n number of elements with replacement
        pairs=[np.zeros((n*n, self.h, self.w,1)) for i in range(2)] #Will create 2 dummy 0 image arrays, each of size n^2

        #print("pairs look like ",pairs[0][1].shape)
        n_t = len(valid_choice)*len(valid_choice)
        if n_t<n:
            targets=np.zeros((n_t*n_t,)) #Placeholder for denoting if the pair have same category or not (0 or 1)
        else:
            targets = np.zeros((n*n,))
        #print("Shape of targets ",targets.shape)
        for i in range(int(len(valid_choice))):
            for j in range(int(len(valid_choice))):
                t1=self.valid_tuple[valid_choice[i]][0][0]
                t2=self.valid_tuple[valid_choice[j]][0][0]
                pairs[0][n*i+j,:,:,:] = self.Xtrain[self.valid_tuple[valid_choice[i]][0][0],self.valid_tuple[valid_choice[i]][1]].reshape(self.w,self.h,1)
                pairs[1][n*i+j,:,:,:] = self.Xtrain[self.valid_tuple[valid_choice[j]][0][0],self.valid_tuple[valid_choice[j]][1]].reshape(self.w,self.h,1)
                if(t1==t2):
                    targets[n*i+j]=1
                else:
                    targets[n*i+j]=0
        return pairs, targets


    def create_support(self,indices):#N controls how many elements you want in the support
        Xval=[]
        Yval=[]
        for i in range(len(indices)):
            Xval.append(self.Xval[indices[i]])
            Yval.append(self.Yval[indices[i]])
        return Xval,Yval #One image from the support set and its true label

    def create_pairs_for_single_test(self,test_img,test_class,indices = []):
        '''
        This function takes an image from the test set and pairs it up with a random arrangement of images from the entire support set (Xval,Yval)
        a custom arrangement can also be passed if indices !=[]
        '''
        if indices==[]:
            indices = rng.choice(int(len(self.Yval)),size=(self.numways,),replace=False)
        supports,classes=self.create_support(indices) #testing 100 way one shot learning
        yes_no=[]
        pairs=[np.zeros((self.numways, self.h, self.w,1)) for i in range(2)]
        for i in range(self.numways):
            #The pairing is happening here
            pairs[0][i,:,:,:]=copy.copy(supports[i])
            pairs[1][i,:,:,:]=copy.copy(test_img)
            if(classes[i]==test_class):
                yes_no.append(1.0) #are the two images similar or dissimilar?
            else:
                yes_no.append(0.0)
        #names are kind of intuitive 
        return pairs,classes,yes_no,indices


    def test_oneshot_ability(self,model,resize_param):
        print("Testing one shot ability ")
        test_set=[]
        occ_acc=0.0
        vac_acc=0.0
        print("time now ",dt.now())
        self.numruns +=1
        '''
        print("Warning ! removing all files in test_result.. wait for validation to finish")
        files = glob.glob('Instances/generalized/cal_data/occupied/*')
        for f in files:
            os.remove(f)
        files = glob.glob('Instances/generalized/cal_data/unoccupied/*')
        for f in files:
            os.remove(f)
        '''
        tpc = 0.0

        p,c,y_n,t_i=self.create_pairs_for_single_test(self.Xtest[0],self.Ytest[0]) #t_i as in true indices in self.Xval
        probs=model.predict(p)
        print("len(probs) ",len(probs))
        confidence_list_occ = np.zeros(len(probs))
        confidence_list_unocc =  np.zeros(len(probs))

        for j in range(len(self.Ytest)): #test on the entire testing set
            p,c,y_n,_=self.create_pairs_for_single_test(self.Xtest[j],self.Ytest[j], indices = t_i) #for a set of tests using the same indices for generating pairs
            probs=model.predict(p)
            cut_num = int(0.50*len(probs))
            pb = np.array(probs)
            pbs = np.sort(pb)
            thresh = pbs[cut_num]
            pred_class=[]
            # Save the bare minimum ground truths also for training
            if self.Ytest[j]==1:
                new_im = Image.fromarray((self.Xtest[j].reshape((resize_param,resize_param))).astype(np.uint8))
                new_im.save('Instances/'+self.filename+'/cal_data/occupied/'+repr(j)+"gt"+'.png')
            if self.Ytest[j]==0:
                new_im = Image.fromarray((self.Xtest[j].reshape((resize_param,resize_param))).astype(np.uint8))
                new_im.save('Instances/'+self.filename+'/cal_data/unoccupied/'+repr(j)+"gt"+'.png')
            for i in range(len(probs)):
                if(probs[i]>=thresh and self.Ytest[j]==1):
                    confidence_list_occ[i] +=1
                if(probs[i]>=thresh and self.Ytest[j]==0):
                    confidence_list_unocc[i] +=1
        correct_hits = 0
        incorrect_hits = 0
        num_add_occ = 0
        num_add_unocc = 0

        #t_i is the true indices
        for i in range(len(probs)):
            if confidence_list_occ[i]>=int(len(self.Ytest)/2):
                
                new_im = Image.fromarray((p[0][i].reshape((resize_param,resize_param))).astype(np.uint8))
                new_im.save('Instances/'+self.filename+'/cal_data/occupied/'+repr(t_i[i])+'class'+repr(c[i][0])+'.png')
                num_add_occ +=1
                self.selected_occ[t_i[i]] += 1
                if c[i]==1:
                    correct_hits +=1
                if c[i]==0:
                    incorrect_hits +=1
            if confidence_list_unocc[i]>=int(len(self.Ytest)/2):
                
                new_im = Image.fromarray((p[0][i].reshape((resize_param,resize_param))).astype(np.uint8))
                new_im.save('Instances/'+self.filename+'/cal_data/unoccupied/'+repr(t_i[i])+'class'+repr(c[i][0])+'.png')
                num_add_unocc +=1
                self.selected_unocc[t_i[i]] += 1
                if c[i]==0:
                    correct_hits +=1
                if c[i]==1:
                    incorrect_hits +=1

        if (correct_hits+incorrect_hits)!=0:
            tpc = float(correct_hits/(correct_hits+incorrect_hits))
            print("Total percentage correct hits",float(correct_hits/(correct_hits+incorrect_hits)))
            self.expected_correct_hits.append(float(correct_hits/(correct_hits+incorrect_hits)))
            print("Expected correct_hits so far ",np.mean(self.expected_correct_hits))
            print("Total images added to training this time",float(correct_hits+incorrect_hits))
            print("Images added to training this time - occupied | unoccupied ",num_add_occ,num_add_unocc)
            if float(correct_hits/(correct_hits+incorrect_hits))>0.5:
                self.numgood +=1
            if float(correct_hits/(correct_hits+incorrect_hits))<0.5:
                self.numbad +=1
        else:
            print("No images were added this time")
        print("Number of good convergences ",self.numgood)
        print("Number of bad convergences ",self.numbad)

        '''
        #saving high accuracy results for visual validation:
        tpc = float(correct_hits/(correct_hits+incorrect_hits))
        if tpc>0.9: #very good!
            print("Warning ! removing all files in visual_validation.. wait to finish")
            files = glob.glob('Instances/'+self.filename+'/visual_validation/occupied/*')
            for f in files:
                os.remove(f)
            files = glob.glob('Instances/'+self.filename+'/visual_validation/unoccupied/*')
            for f in files:
                os.remove(f)

            for i in range(len(probs)):
                if confidence_list_occ[i]>=int(len(self.Ytest)/2):
                    new_im = Image.fromarray((p[0][i].reshape((resize_param,resize_param))).astype(np.uint8))
                    new_im.save('Instances/'+self.filename+'/visual_validation/occupied/'+repr(t_i[i])+'class'+repr(c[i][0])+'.png')
                    
                if confidence_list_unocc[i]>=int(len(self.Ytest)/2):
                    new_im = Image.fromarray((p[0][i].reshape((resize_param,resize_param))).astype(np.uint8))
                    new_im.save('Instances/'+self.filename+'/visual_validation/unoccupied/'+repr(t_i[i])+'class'+repr(c[i][0])+'.png')
            print("Exiting program for user to check")
            sys.exit(0)
        '''

        o = 0.0
        u = 0.0
        docc = 0
        dunocc = 0
        confusion = []
        for i in range(len(self.selected_occ)):
            if self.selected_occ[t_i[i]]>self.selected_unocc[t_i[i]] and self.Yval[t_i[i]]==1:
                o+=1
            if self.selected_occ[t_i[i]]<self.selected_unocc[t_i[i]] and self.Yval[t_i[i]]==0:
                u+=1
            if self.selected_occ[t_i[i]]>self.selected_unocc[t_i[i]]: #this condition means the NN says that most likely the image is occupied
                docc+=1
            if self.selected_occ[t_i[i]]<self.selected_unocc[t_i[i]]:
                dunocc+=1

            confusion.append((1+self.selected_occ[t_i[i]])/(1+self.selected_unocc[t_i[i]]))
        
        print("Average confusion ",np.mean(confusion))
        if docc==0.0 or dunocc==0.0:
            print("Division by 0 will occur !")
            docc+=1
            dunocc+=1
        print("Expected clustering accuracy on support set occ|unocc ",float(o/docc),float(u/dunocc))
        print("Total number of images selected last 10time occ|unocc ",docc,dunocc)
        self.writelog([float(correct_hits/(correct_hits+incorrect_hits)),np.mean(self.expected_correct_hits),float((correct_hits+incorrect_hits)),num_add_occ, num_add_unocc, self.numgood,self.numbad,float(o/docc),float(u/dunocc)])

        self.writeNthlog([self.numruns, np.mean(self.expected_correct_hits),docc,dunocc, docc+dunocc, float(o/docc),float(u/dunocc)])
        
        return np.round(tpc,1)

