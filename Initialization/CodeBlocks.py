#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:57:31 2021

@author: emanuele
"""

#for nn
import torch
import numpy as np
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
#import tensorboard
#from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder


import wandb

from collections import Counter



# for data visualization
import matplotlib.pyplot as plt

#to set network architecture
import torch.nn as nn
import torch.nn.functional as F

import psutil
from inspect import currentframe

#Whatever you want to do with shells like running an application, copying files etc., you can do with subprocess. It has run function which does it for you!
import subprocess 
"""
#accelerate the training
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
"""

#packages iclude by default (don't have to manually install in your virtual env)

#manage the input of values from outside the script 
import argparse
#manage the folder creation
import os
import time

#deep copy mutable object
import copy

import math

import random


#for pre-trained architectures
import torchvision.models as models
import torchvision.transforms as transforms


torch.set_printoptions(precision=17)

#%% CHECKING CLASS
#load the dataset and compute mean and std for the standardization of its element (by now only implemented for cifar10 dataset)
#torch.set_default_tensor_type(torch.DoubleTensor)


class DatasetMeanStd:
    
    def __init__(self, params):
        """
        This class is a tool to compute mean and std to standardise (or check) your dataset
        the init function load the training and test dataset 
        Parameters
        ----------
        DatasetName : string
        this is a string that encode the dataset that will be used

        Returns
        -------
        None.

        """
        
        self.params = params.copy()

        self.DatasetName = self.params['Dataset']
        self.ClassesList = self.params['label_map'].keys() #get the list of images in the dataset in case of subset of classes
        self.transform = transforms.ToTensor()
        if(self.DatasetName=='CIFAR10'):
            self.train_data = datasets.CIFAR10(root = self.params['DataFolder'], train = True, download = False, transform = self.transform)
            self.test_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = False, transform = self.transform)
        elif(self.DatasetName=='CIFAR100'):
            self.train_data = datasets.CIFAR100(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
            
        elif(self.DatasetName=='MNIST'):
            self.train_data = datasets.MNIST(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.MNIST(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
        else:
            self.train_data = ImageFolder(self.params['DataFolder']+'/train')
            self.test_data = ImageFolder(self.params['DataFolder']+'/test')
            
            
    #TODO: include also the option to calculate mean and std for test sets and MNIST dataset
        i=0
        for item in self.train_data:
            if i==0:
                print('image', item[0].size())
                print('label', item[1])
                i+=1
                
    def Mean(self):
        """
        Compute the mean of the dataset (only Cifar10 for now) for the standardization (image vectors normalization)
        Returns
        -------
        list
            mean value for each channel

        """
        
        
        
        if (self.DatasetName == 'CIFAR10') or (self.DatasetName == 'CIFAR100'):
            imgs = [item[0] for item in self.train_data if item[1] in self.ClassesList]  # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate mean over each channel (r,g,b)
            mean_r = imgs[:,0,:,:].mean()
            mean_g = imgs[:,1,:,:].mean()
            mean_b = imgs[:,2,:,:].mean()   

            return (mean_r, mean_g, mean_b)

        elif (self.DatasetName == 'MNIST'):
            imgs = [item[0] for item in self.train_data if item[1] in self.ClassesList]  # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            # calculate mean over each channel (r,g,b)
            mean = imgs[:,0,:,:].mean()

            return mean        

    def Std(self):
        
        """
        Compute the std of the dataset (only Cifar10 for now) for the standardization (image vectors normalization)

        Returns
        -------
        list
            std value for each channel

        """
        
        if (self.DatasetName == 'CIFAR10') or (self.DatasetName == 'CIFAR100'):
            imgs = [item[0] for item in self.train_data] # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate std over each channel (r,g,b)
            std_r = imgs[:,0,:,:].std()
            std_g = imgs[:,1,:,:].std()
            std_b = imgs[:,2,:,:].std()  
            
            return(std_r, std_g, std_b)

        elif (self.DatasetName == 'MNIST'):
            imgs = [item[0] for item in self.train_data] # item[0] and item[1] are image and its label
            imgs = torch.stack(imgs, dim=0).numpy()
            
            # calculate std over each channel (r,g,b)
            std = imgs[:,0,:,:].std()
 
            
            return std            
            


#%% DEFINING TIMES CLASS
class Define:
    def __init__(self, params, n_epochs, NSteps, NBatches, StartPoint, PreviousTimes):
        """
        This class contain methods to create list of logarithmically equispced times, and the ones for the correlations computation


        Parameters
        ----------
        params : dict 
            store all the relevant parameter defined in the main code
        n_epochs : int
            express the total number of epoches
        NSteps : int
            express the number of the final time arrow (this arrow will be used to set the times where to evaluate state of the training (store train/valuation/(test) performances))
        NBatches : int
            Number of batches in the dataset (fixed by the dataset size and batch size)
        StartPoint : int
            starting point for the beginning of the simulation (different for new/retrieved runs)
        PreviousTimes : array
            array of previous times to be attached (in the RETRIEVE mode) at the beginning of the time vector

        Returns
        -------
        None.

        """
        self.params = params.copy()
        self.n_epochs = n_epochs
        self.NSteps = NSteps
        self.NBatches = NBatches
        self.PreviousTimes = PreviousTimes
            
        #we differenciate the 2 cases (because we cannot use as start point log(0))
        if (self.params['StartMode']=='BEGIN'):
            self.StartPoint = StartPoint #in this case we substitute into np.logspace() (below) directly the input (since it is 0) 2**0=1
        if (self.params['StartMode']=='RETRIEVE'):
            self.StartPoint = np.log2(StartPoint) #here instead we substitute into np.logspace() (below) the log(input), where the input correspond to the end (MaxStep) of the last simulation 
        
    #this method return an array of Nsteps dimension with logaritmic equispaced steps : we use it for stocastic algortim as SGD and PCNSGD
    def StocasticTimes(self):
        '''
        define the time time vector for the evaluation stops (stochastic case)
        NOTE: in case of RETRIEVE mode the new times will be equispaced but, in general, with a different spacing between consecutive times with respect to the old vector
        Returns
        -------
        Times : numpy array
            return the logarithmic equispaced steps for stochastic algorithms (as SGD).

        '''    
        MaxStep = self.n_epochs*self.NBatches #the last factor is due to the fact that in the PCNSGD we trash some batches (so the number of steps is slightly lower); 0.85 is roughtly set thinking that batch size, also in the unbalance case will be chosen to not esclude more than 15% of batches
        Times = np.logspace(self.StartPoint, np.log2(MaxStep),num=self.NSteps, base=2.) 
        Times = np.rint(Times).astype(int)   
        
        if (self.params['StartMode']=='BEGIN'):
            if self.NSteps>4:
                for ind in range(0,4): #put the initial times linear to store initial state
                    Times[ind] = ind+1    
        
        for steps in range (0, self.NSteps-1):
            while Times[steps] >= Times[steps+1]:
                Times[steps+1] = Times[steps+1]+1
        
        if (self.params['StartMode']=='RETRIEVE'): #in case of continuing previous simulation we concatenate the new times sequence at the end of the old one
            Times = np.concatenate((self.PreviousTimes,Times[1:]), axis=0)


        #here we just reproduce the same time vector to compare the single piece simulation withe the interrupt one (to test the checkpoint)
        if (self.params['CheckMode']=='ON'):
            MaxStep = self.params['n_epochsComp']*self.NBatches
            Times = np.logspace(0, np.log2(MaxStep),num=self.params['NStepsComp'], base=2.) 
            Times = np.rint(Times).astype(int)
            if self.NSteps>4:
                for ind in range(0,4): #put the initial times linear to store initial state
                    Times[ind] = ind+1    
            
            for steps in range (0, self.NSteps-1):
                while Times[steps] >= Times[steps+1]:
                    Times[steps+1] = Times[steps+1]+1
                
        return Times
          
        
 
    #if we are using a full batch approach we fix the equispaced times with the numbers of epoches
    def FullBatchTimes(self):       
        '''
        define the time vector for the evaluation stops (full batch case)
        NOTE: in case of RETRIEVE mode the new times will be equispaced but, in general, with a different spacing between consecutive times with respect to the old vector
        Returns
        -------
        Times : numpy array
            return the logarithmic equispaced steps for full batch algorithms (as SGD)..

        '''
        MaxStep = self.n_epochs
        Times = np.logspace(self.StartPoint, np.log2(MaxStep),num=self.NSteps, base=2.) 
        Times = np.rint(Times).astype(int)

        if (self.params['StartMode']=='BEGIN'):    
            if self.NSteps>4:
                for ind in range(0,4): #put the initial times linear to store initial state
                    Times[ind] = ind+1    
        
        for steps in range (0, self.NSteps-1):
            while Times[steps] >= Times[steps+1]:
                Times[steps+1] = Times[steps+1]+1

        if (self.params['StartMode']=='RETRIEVE'): #in case of continuing previous simulation we concatenate the new times sequence at the end of the old one
            print('previous times are ', self.PreviousTimes, 'new ones ',  Times)
            Times = np.concatenate((self.PreviousTimes,Times[1:]), axis=0)                
            print('the- concatenation of the 2 ', Times)
            
            
        #here we just reproduce the same time vector to compare the single piece simulation withe the interrupt one (to test the checkpoint)
        if (self.params['CheckMode']=='ON'):
            MaxStep = self.params['n_epochsComp']
            Times = np.logspace(0, np.log2(MaxStep),num=self.params['NStepsComp'], base=2.) 
            Times = np.rint(Times).astype(int)
       
            if self.NSteps>4:
                for ind in range(0,4): #put the initial times linear to store initial state
                    Times[ind] = ind+1    
            
            for steps in range (0, self.NSteps-1):
                while Times[steps] >= Times[steps+1]:
                    Times[steps+1] = Times[steps+1]+1
            
            
        return Times

    
    #TODO: adapt correlation times to the logic of RETRIEVE
    #setting the times for the computation of correlations:
    #i set tw log-equispaced on the bigger interval possible and t log equispaced in a sub interval whose size is under the difference (MaxStep-last tw)
    def CorrTimes(self, Ntw, Nt, tw, t):
        """
        This method combine the 2 vectors (tw and t) into a 2-D matrix to express correlation times
        NOTE: correlation measure on the last version is not stable yet
        Parameters
        ----------
        Ntw : int
            number of first times for the correlation computation
        Nt : int
            number of second times for the correlation computation.
        MaxStep : int
            Number of maximum steps associated with the fixed number of epoches of the simulation.
        tw : array 
            array of starting points.
        t : array
            array of second points for the 2 point correlation computation.

        Returns
        -------
        CorrTimes : 2-D array
            matrix of 2-point correlation times.

        """
        
        
        #defining the correlation times matrix
        CorrTimes = np.zeros((Ntw, Nt))
        
        for i in range(0, Ntw):
            for j in range(0, Nt):
                CorrTimes[i][j] = tw[i] + t[j]
        return CorrTimes
    
    #correlation are calculated between 2 times, this function return the list of starting times
    def FirstCorrTimes(self, Ntw, MaxStep):
        """
        Create the vector of starting points for the 2-point correlation computation
        
        Parameters
        ----------
        Ntw : int
            number of first times for the correlation computation.
        MaxStep : int
            Number of maximum steps associated with the fixed number of epoches of the simulation..

        Returns
        -------
        None.

        """
        
        tw =  np.logspace(self.StartPoint, np.log2(MaxStep*0.5),num=(Ntw +1), base=2.) 
        tw = np.rint(tw).astype(int)
        #shift forward the equal sizes
        for steps in range (0, Ntw):
            while tw[steps] >= tw[steps+1]:
                tw[steps+1] = tw[steps+1]+1
        
        return tw
        
    #correlation are calculated between 2 times, this function return the list of arriving (second) times
    def SecondCorrTimes(self,Ntw, Nt, tw, MaxStep, spacing_mode = 'log'):     
        """
        Create the vector of second points for the 2-point correlation computation

        Parameters
        ----------
        Ntw : int
            number of first times for the correlation computation
        Nt : int
            number of second times for the correlation computation.

        tw : array 
            array of starting points.
            
        MaxStep : int
            Number of maximum steps associated with the fixed number of epoches of the simulation.
            
        spacing_mode: string
            the spacing mode: linear o logharithmic (default)
        Returns
        -------
        None.

        """
        if (spacing_mode == 'log'):
            #saving 0 and 1 as first ts to evaluate correlation and overlap with the same config. and with the time soon after

            t = np.logspace(3, np.log2(MaxStep-tw[Ntw]),num=(Nt +1), base=2.) 
            t = np.rint(t).astype(int)
            t[0] = 0
            t[1] = 1
            
        elif(spacing_mode == 'linear'):

            t = np.linspace(2, MaxStep-tw[Ntw], num = Nt+1, dtype = int)
            t[0] = 0
            t[1] = 1
        else:
            print('Invalid spacing_mode given as input to SecondCorrTimes function', file=self.params['WarningFile'])
            
        #shift forward the equal sizes
        for steps in range (0, Nt):
            while t[steps] >= t[steps+1]:
                t[steps+1] = t[steps+1]+1
        return t
           

#%% VARIABLE CLASS
#start creating a class which contain all variable used in the others, so these last ones can inherethe them
class NetVariables:
    """
    This class is the container of all the relevant variables that will be stored on files
    """
    #we initialize variable using none as default value to set empty variable because is not good to use directly [] (mutable object in jeneral as default value (deepen this concept)) 
    #so we set None as default value for all variables we don't want to pass
    def __init__(self, params, TrainLoss = None, TestLoss = None, ValidLoss = None , TrainAcc = None ,TestAcc = None, ValidAcc = None , WeightNorm = None, GradientNorm = None,
                 TP = None, TN = None, FP = None, FN = None, TotP = None, Prec = None, Recall = None, FMeasure = None, TrainAngles = None,
                 PCGAngles = None, GradAnglesNormComp = None, StepGradientClassNorm = None, TrainClassesLoss = None, TrainClassesAcc = None, TestAngles = None, TestClassesLoss = None, TestClassesAcc = None, ValidClassesLoss = None, ValidClassesAcc = None,
                 RepresentationClassesNorm = None, ClassesGradientNorm = None,  
                 TwWeights = None, TwoTimesOverlap= None, TwoTimesDistance=None ):
                 
            self.num_classes = params['n_out']
            #self.NSteps = params['NSteps']
            #if you want to have measures at each step (not only the ones corresponding to "Times"' list we modify the above line to modify the shape of the corresponding arraies)
            self.NSteps = params['NSteps']
            self.n_epochs = params['n_epochs']
            self.Ntw = params['Ntw']
            self.Nt = params['Nt']
            

            #prepare variable for accuracy and loss plot
            if TrainLoss is None:
                self.TrainLoss = []
            else:
                 self.TrainLoss = TrainLoss
                 
            if TestLoss is None:
                self.TestLoss = []
            else:
                 self.TestLoss = TestLoss            
            
            if TrainAcc is None:
                self.TrainAcc = []
            else:
                 self.TrainAcc = TrainAcc            
            
            if TestAcc is None:
                self.TestAcc = []
            else:
                 self.TestAcc = TestAcc    

            if ValidLoss is None:
                self.ValidLoss = []
            else:
                 self.ValidLoss = ValidLoss            
            
            if ValidAcc is None:
                self.ValidAcc = []
            else:
                 self.ValidAcc = ValidAcc        

            
            if WeightNorm is None: #MEASURE TO ADD TO RECALL LOGIC
                self.WeightNorm = []
            else:
                 self.WeightNorm = WeightNorm    
            
            if GradientNorm is None: #MEASURE TO ADD TO RECALL LOGIC
                self.GradientNorm = []            
            else:
                 self.GradientNorm = GradientNorm    
            
            #DEFINE SOME VARIABLE CONTAINERS
            #vector for true positive and false positive (for the precision) 
            if TP is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TP = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.TP = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.TP = TP    
            
            if TN is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TN = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.TN = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.TN = TN    
            
            if FP is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.FP = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.FP = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.FP = FP    
            
            if FN is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.FN = torch.zeros((self.num_classes , (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.FN = torch.zeros((self.num_classes , (self.NSteps)))
            else:
                 self.FN = FN    
            
            if TotP is None:
                self.TotP = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.TotP = TotP    
            
            if Prec is None:
                self.Prec = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.Prec = Prec    
            
            if Recall is None:
                self.Recall = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.Recall = Recall    
            
            if FMeasure is None:
                self.FMeasure = torch.zeros((self.num_classes , self.NSteps))
            else:
                 self.FMeasure = FMeasure    
            
            #angules variables
            #NOTE: IT IS IMPORTANT TO FILL THE ARRAY WITH THE CORRECT TYPE, BECAUSE THIS WILL BE THE DEFAULT DATA TYPE OF THE ARRAY USED; TOSPECIFY EXPLICITLY THE DATA TYPE USE THE OPTION dtype=...
            #we add one more component (n_epochs + 1 instead of n_epochs) because we also store the starting state to check the randomness of representation state at the beginning
            if TrainAngles is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TrainAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps + 1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.TrainAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps)), 1000.)                
            else:
                 self.TrainAngles = TrainAngles    
            
            if PCGAngles is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.PCGAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps + 1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'): #restarting in RETRIEVE mode we don't need the measure at step 0 (before the training start)
                    self.PCGAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps)), 1000.)
            else:
                 self.PCGAngles = PCGAngles   
                 
            if GradAnglesNormComp is None:
                self.GradAnglesNormComp = np.full((int(self.num_classes*(self.num_classes-1)/2), (self.NSteps)), 1000.)
            else:
                self.GradAnglesNormComp = GradAnglesNormComp
                
                 
            if StepGradientClassNorm is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.StepGradientClassNorm = np.full((self.num_classes, (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.StepGradientClassNorm = np.full((self.num_classes, (self.NSteps)), 1000.)
            else:   
                self.StepGradientClassNorm = StepGradientClassNorm
            
            if TrainClassesLoss is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TrainClassesLoss = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TrainClassesLoss = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.TrainClassesLoss = TrainClassesLoss    
            
            if TrainClassesAcc is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TrainClassesAcc = np.full((int(self.num_classes), (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TrainClassesAcc = np.full((int(self.num_classes), (self.NSteps)), 1000.)
            else:
                 self.TrainClassesAcc = TrainClassesAcc    
                 
            #same measures for test set
            if TestAngles is None:
                self.TestAngles = np.full((int(self.num_classes*(self.num_classes-1)/2), self.NSteps), 10.)
            else:
                 self.TestAngles = TestAngles    
            
            if TestClassesLoss is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TestClassesLoss = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TestClassesLoss = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.TestClassesLoss = TestClassesLoss    
            
            if TestClassesAcc is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.TestClassesAcc = np.full((int(self.num_classes), (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.TestClassesAcc = np.full((int(self.num_classes), (self.NSteps)), 1000.)
            else:
                 self.TestClassesAcc = TestClassesAcc    
            
            if ValidClassesLoss is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.ValidClassesLoss = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.ValidClassesLoss = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.ValidClassesLoss = ValidClassesLoss  

            if ValidClassesAcc is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.ValidClassesAcc = np.full((int(self.num_classes), (self.NSteps+1)), 1000.)
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.ValidClassesAcc = np.full((int(self.num_classes), (self.NSteps)), 1000.)
            else:
                 self.ValidClassesAcc = ValidClassesAcc             
            
            if RepresentationClassesNorm is None:
                if (self.params['StartMode']=='BEGIN'):
                    self.RepresentationClassesNorm = np.zeros((self.num_classes, (self.NSteps + 1)))
                elif (self.params['StartMode']=='RETRIEVE'):
                    self.RepresentationClassesNorm = np.zeros((self.num_classes, (self.NSteps)))
            else:
                 self.RepresentationClassesNorm = RepresentationClassesNorm    
            
            if ClassesGradientNorm is None:
                self.ClassesGradientNorm = np.zeros((self.n_epochs, self.num_classes))
            else:
                 self.ClassesGradientNorm = ClassesGradientNorm    
            
   

            #define the list where we will put the copies of the weights
            if (params['SphericalConstrainMode']=='ON'):
                if TwWeights is None:     
                    self.TwWeights = []
                if TwoTimesOverlap is None:     
                    self.TwoTimesOverlap = torch.zeros((self.Ntw, self.Nt))       
                if TwoTimesDistance is None:     
                    self.TwoTimesDistance = torch.zeros((self.Ntw, self.Nt))  
        







#%% USEFUL METHOD FOR THE NETWORK CLASSES
#HERE WE DEFINE A CLASSES THAT COINTAIN USEFUL TOOL FOR THE NET CLASSES


class OrthoInit:
    """
    #ORTHOGONAL CONDITION: since we are using a more extended CNN we add the possibility to initialize the weight according to the method proposed in the following article:
    #Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks
    #below some methods used for this purpose  
    """
    def __init__(self):
        pass
    
    ######################################Generating 2D orthogonal initialization kernel####################################
    #generating uniform orthogonal matrix
    def _orthogonal_matrix(self, dim):
        a = torch.zeros((dim, dim)).normal_(0, 1)
        q, r = torch.linalg.qr(a)
        d = torch.diag(r, 0).sign()
        diag_size = d.size(0)
        d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
        q.mul_(d_exp)
        return q
    
    #generating orthogonal projection matrix,i.e. the P,Q of Algorithm1 in the original
    def _symmetric_projection(self, n):
        """Compute a n x n symmetric projection matrix.
        Args:
          n: Dimension.
        Returns:
          A n x n orthogonal projection matrix, i.e. a matrix P s.t. P=P*P, P=P^T.
        """
        q = self._orthogonal_matrix(n)
        # randomly zeroing out some columns
        # mask = math.cast(random_ops.random_normal([n], seed=self.seed) > 0,
        # #                      self.dtype)
        mask = torch.randn(n)
    
        c = torch.mul(mask,q)
        U,_,_= torch.svd(c)
        U1 = U[:,0].view(len(U[:,0]),1)
        P = torch.mm(U1,U1.t())
        P_orth_pro_mat = torch.eye(n)-P
        return P_orth_pro_mat
    
    #generating block matrix the step2 of the Algorithm1 in the original
    def _block_orth(self, p1, p2):
        """Construct a 2 x 2 kernel. Used to construct orthgonal kernel.
        Args:
          p1: A symmetric projection matrix (Square).
          p2: A symmetric projection matrix (Square).
        Returns:
          A 2 x 2 kernel [[p1p2,         p1(1-p2)],
                          [(1-p1)p2, (1-p1)(1-p2)]].
        Raises:
          ValueError: If the dimensions of p1 and p2 are different.
        """
        if p1.shape != p2.shape:
            raise ValueError("The dimension of the matrices must be the same.")
        kernel2x2 = {}#Block matrices are contained by a dictionary
        eye = torch.eye(p1.shape[0])
        kernel2x2[0, 0] = torch.mm(p1, p2)
        kernel2x2[0, 1] = torch.mm(p1, (eye - p2))
        kernel2x2[1, 0] = torch.mm((eye - p1), p2)
        kernel2x2[1, 1] = torch.mm((eye - p1), (eye - p2))
    
        return kernel2x2
    
    #compute convolution operator of equation2.17 in the original
    def _matrix_conv(self, m1, m2):
        """Matrix convolution.
        Args:
          m1: A k x k dictionary, each element is a n x n matrix.
          m2: A l x l dictionary, each element is a n x n matrix.
        Returns:
          (k + l - 1) * (k + l - 1) dictionary each element is a n x n matrix.
        Raises:
          ValueError: if the entries of m1 and m2 are of different dimensions.
        """
    
        n = m1[0, 0].shape[0]
        if n != m2[0, 0].shape[0]:
            raise ValueError("The entries in matrices m1 and m2 "
                             "must have the same dimensions!")
        k = int(np.sqrt(len(m1)))
        l = int(np.sqrt(len(m2)))
        result = {}
        size = k + l - 1
        # Compute matrix convolution between m1 and m2.
        for i in range(size):
            for j in range(size):
                result[i, j] = torch.zeros(n,n)
                for index1 in range(min(k, i + 1)):
                    for index2 in range(min(k, j + 1)):
                        if (i - index1) < l and (j - index2) < l:
                            result[i, j] += torch.mm(m1[index1, index2],
                                                            m2[i - index1, j - index2])
        return result
    
    def _dict_to_tensor(self, x, k1, k2):
        """Convert a dictionary to a tensor.
        Args:
          x: A k1 * k2 dictionary.
          k1: First dimension of x.
          k2: Second dimension of x.
        Returns:
          A k1 * k2 tensor.
        """
        return torch.stack([torch.stack([x[i, j] for j in range(k2)])
                                for i in range(k1)])
    
    #generating a random 2D orthogonal Convolution kernel
    def _orthogonal_kernel(self, tensor):
        """Construct orthogonal kernel for convolution.
        Args:
          ksize: Kernel size.
          cin: Number of input channels.
          cout: Number of output channels.
        Returns:
          An [ksize, ksize, cin, cout] orthogonal kernel.
        Raises:
          ValueError: If cin > cout.
        """
        ksize = tensor.shape[2]
        cin = tensor.shape[1]
        cout = tensor.shape[0]
        if cin > cout:
            raise ValueError("The number of input channels cannot exceed "
                             "the number of output channels.")
        orth = self._orthogonal_matrix(cout)[0:cin, :]#这就是算法1中的H
        if ksize == 1:
            return torch.unsqueeze(torch.unsqueeze(orth,0),0)
    
        p = self._block_orth(self._symmetric_projection(cout),
                             self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(self._symmetric_projection(cout),
                                    self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            for j in range(ksize):
                p[i, j] = torch.mm(orth, p[i, j])
        tensor.copy_(self._dict_to_tensor(p, ksize, ksize).permute(3,2,1,0))
        return tensor
    
    #defining 2DConvT orthogonal initialization kernel
    def ConvT_orth_kernel2D(self,tensor):
        ksize = tensor.shape[2]
        cin = tensor.shape[0]
        cout = tensor.shape[1]
        if cin > cout:
            raise ValueError("The number of input channels cannot exceed "
                             "the number of output channels.")
        orth = self._orthogonal_matrix(cout)[0:cin, :]  # 这就是算法1中的H
        if ksize == 1:
            return torch.unsqueeze(torch.unsqueeze(orth, 0), 0)
    
        p = self._block_orth(self._symmetric_projection(cout),
                        self._symmetric_projection(cout))
        for _ in range(ksize - 2):
            temp = self._block_orth(self._symmetric_projection(cout),
                               self._symmetric_projection(cout))
            p = self._matrix_conv(p, temp)
        for i in range(ksize):
            for j in range(ksize):
                p[i, j] = torch.mm(orth, p[i, j])
        tensor.copy_(self._dict_to_tensor(p, ksize, ksize).permute(2, 3, 1, 0))
        return tensor
    #Call method
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.shape[0] > m.weight.shape[1]:
                    self._orthogonal_kernel(m.weight.data)
                    m.bias.data.zero_()
                else:
                    nn.init.orthogonal_(m.weight.data)
                    m.bias.data.zero_()
    
            elif isinstance(m, nn.ConvTranspose2d):
                if m.weight.shape[1] > m.weight.shape[0]:
                    self.ConvT_orth_kernel2D(m.weight.data)
                   # m.bias.data.zero_()
                else:
                    nn.init.orthogonal_(m.weight.data)
                   # m.bias.data.zero_()
    
               # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
    '''
    Algorithm requires The number of input channels cannot exceed the number of output channels.
     However, some questions may be in_channels>out_channels. 
     For example, the final dense layer in GAN. If counters this case, Orthogonal_kernel is replaced by the common orthogonal init'''
    '''
    for example,
    net=nn.Conv2d(3,64,3,2,1)
    net.apply(Conv2d_weights_orth_init)
    '''
    
    def makeDeltaOrthogonal(self, in_channels=3, out_channels=64, kernel_size=3, gain=torch.Tensor([1])):
        weights = torch.zeros(out_channels, in_channels, kernel_size, kernel_size)
        out_channels = weights.size(0)
        in_channels = weights.size(1)
        if weights.size(1) > weights.size(0):
            raise ValueError("In_filters cannot be greater than out_filters.")
        q = self._orthogonal_matrix(out_channels)
        q = q[:in_channels, :]
        q *= torch.sqrt(gain)
        beta1 = weights.size(2) // 2
        beta2 = weights.size(3) // 2
        weights[:, :, beta1, beta2] = q
        return weights
    #Calling method is the same as the above _orthogonal_kernel
    ######################################################END###############################################################



#%% DEFINE NN ARCHITECTURE

class Single_HL(nn.Module, NetVariables):
    def __init__(self, params):
        """
        Network class: this is a simple toy network  ()MLP with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below

        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code

        Returns
        -------
        None.

        """
        #super(Net,self).__init__()
        #super(Net, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)        
        
        
        
        # number of hidden nodes in each layer (512)
        hidden_1 = self.params['HiddenLayerNumNodes']
        
        maxpooled_hidden = math.ceil(self.params['HiddenLayerNumNodes']/self.params['MaxPoolArgs']['kernel_size']) #dimension after max pooling; if you want to avoid max pooling just set stride to 1

        # linear layer (784 -> hidden_1)
        if  (params['Dataset']=='MNIST'):
            self.fc1 = nn.Linear(28*28, hidden_1)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (params['Dataset']=='CIFAR10' or params['Dataset']=='GaussBlob'):
            self.fc1 = nn.Linear(32*32*3, hidden_1)
        # linear layer (n_hidden -> 10)

        self.layer1 = nn.Sequential(
            nn.ReLU() ,
            nn.MaxPool1d(kernel_size=self.params['MaxPoolArgs']['kernel_size'])
            )
        
        self.fc2 = nn.Linear(maxpooled_hidden, 1)
        
        #weights initialization (this step can also be put below in a separate def)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        #initialize the bias to 0
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
    #I return from the forward a dictionary to get the output after each layer not only the last one
    #this is useful for example in the inter-classes angles (calculated throught the scalar product between inner layers representation)
    def forward(self,x):
        
        outs = {}
        # flatten image input
        if  (self.params['Dataset']=='MNIST' ):
            x = x.view(-1,28*28)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (self.params['Dataset']=='CIFAR10' or self.params['Dataset']=='GaussBlob'):
            x = x.view(-1,32*32*3)
        # add hidden layer, with relu activation function
        Fc1 = self.fc1(x)
        
        #print('prima del pool')
        #print(Fc1)
        
        #Fc1 = self.layer1(Fc1)

        #print('dopo del pool')
        #print(Fc1)
        
        
        outs['l2'] = Fc1
        
        # add hidden layer, with relu activation function
        #Fc2 = F.relu(self.fc2(Fc1))
        #if you want to use a tanh activation function don't use F.tanh wich is deprecated; instead substitute the line above with the one below
        Fc2 = (self.fc2(Fc1))
        
        
        # add output layer
        outs['out'] = Fc2
        
        #print(torch.sign(Fc2).int())
        if self.params['SignCountFlag']=='ON': #this condition doesn't work because self.params is defined in this class during init and then stay fixed; to make it work you should sent the parameter as an explicit forward input
            for key in self.params['SignProp']:
                self.params['SignProp'][key]+=torch.sum((torch.sign(Fc2).int()==key))
                #print(key, torch.sum((torch.sign(Fc2).int()==key)))
        
        outs['pred'] = torch.tensor([self.params['sign_to_label_dict'][x.item()] for x in torch.sign(Fc2).int()]).to(self.params['device']) 

        
        #print(torch.sign(Fc2).int(), flush= True)
        #print(outs['pred'], flush= True)
        
        return outs



class MC_Single_HL(nn.Module, NetVariables):
    def __init__(self, params):
        """
        Network class: this is a simple toy network  ()MLP with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below

        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code

        Returns
        -------
        None.

        """
        #super(Net,self).__init__()
        #super(Net, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)        
        
        
        
        # number of hidden nodes in each layer (512)
        hidden_1 = self.params['HiddenLayerNumNodes']
        
        maxpooled_hidden = math.ceil(self.params['HiddenLayerNumNodes']/self.params['MaxPoolArgs']['kernel_size']) #dimension after max pooling; if you want to avoid max pooling just set stride to 1

        # linear layer (784 -> hidden_1)
        if  (params['Dataset']=='MNIST'):
            self.fc1 = nn.Linear(28*28, hidden_1)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (params['Dataset']=='CIFAR10' or params['Dataset']=='GaussBlob'):
            self.fc1 = nn.Linear(32*32*3, hidden_1)
        # linear layer (n_hidden -> 10)

        self.layer1 = nn.Sequential(
            nn.ReLU() ,
            nn.MaxPool1d(kernel_size=self.params['MaxPoolArgs']['kernel_size'])
            )
        
        self.fc2 = nn.Linear(maxpooled_hidden, self.num_classes)
        
        print('class number', self.num_classes)
        
        #weights initialization (this step can also be put below in a separate def)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        #initialize the bias to 0
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
    #I return from the forward a dictionary to get the output after each layer not only the last one
    #this is useful for example in the inter-classes angles (calculated throught the scalar product between inner layers representation)
    def forward(self,x):
        
        outs = {}
        # flatten image input
        if  (self.params['Dataset']=='MNIST' ):
            x = x.view(-1,28*28)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (self.params['Dataset']=='CIFAR10' or self.params['Dataset']=='GaussBlob'):
            x = x.view(-1,32*32*3)
        # add hidden layer, with relu activation function
        Fc1 = self.fc1(x)
        
        #we have then different option depending on what configuration (e.g. activation function) we want to employ: the selection is triggered by params NetConf
        
        
        #print('prima del pool') 
        #print(Fc1)  # print to check
        if self.params['NetConf']=='RelU+MaxPool' : #setting self.params['MaxPoolArgs']['kernel_size'] we fall into the specific case of only ReLU
            Fc1 = self.layer1(Fc1)

        #print('dopo del pool')
        #print(Fc1) #print to check
        
        
        outs['l2'] = Fc1
        
        # add hidden layer, with relu activation function
        #Fc2 = F.relu(self.fc2(Fc1))
        
        
        Fc2 = (self.fc2(Fc1))
        
        
        # add output layer
        outs['out'] = Fc2
        
        
        #print('f(x)', Fc2)
        
        outs['pred'] = torch.argmax(Fc2, dim=1)

        #print('pred', outs['pred'])
        
        #print(torch.sign(Fc2).int(), flush= True)
        #print(outs['pred'], flush= True)
        
        return outs


#define shifted ReLU activation function

def ShiftedReLU(x):
    m = nn.ReLU()
    return -(1./math.sqrt(math.pi)) + m(x) #we subtract to the node (after passed throught relu act. funct.) the mean value of its distribution

class No_IGB_ReLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return ShiftedReLU(input) # simply apply already implemented SiLU


class Deep_HL(nn.Module, NetVariables):
    def __init__(self, params):
        """
        Network class: this is a simple toy network  ()MLP with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below

        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code

        Returns
        -------
        None.

        """
        #super(Net,self).__init__()
        #super(Net, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)        
        
    
        
        # number of hidden nodes in each layer (512)
        
        maxpooled_hidden = math.ceil(self.params['HiddenLayerNumNodes'][-1]/self.params['MaxPoolArgs']['kernel_size']) #dimension after max pooling; if you want to avoid max pooling just set stride to 1

        #create an instance of new class of customized activation function (to initialize it)
        self.Null_Mean_ReLU = No_IGB_ReLU()
        
        

        # linear layer (784 -> hidden_1)
        if  (params['Dataset']=='MNIST'):
            input_dim = 28*28
            self.params['HiddenLayerNumNodes'].insert(0, input_dim)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (params['Dataset']=='CIFAR10' or params['Dataset']=='GaussBlob'):
            input_dim = 32*32*3
            self.params['HiddenLayerNumNodes'].insert(0, input_dim)
        # linear layer (n_hidden -> 10)

        self.layer1 = nn.Sequential(
            nn.ReLU() ,
            nn.MaxPool1d(kernel_size=self.params['MaxPoolArgs']['kernel_size'])
            )
                

        #I return from the forward a dictionary to get the output after each layer not only the last one
        #this is useful for example in the inter-classes angles (calculated throught the scalar product between inner layers representation)
    
    
        #self.features = self._make_layers(cfg[vgg_name])
        self.features = self._make_layers() #TODO:  SUBSTITUTE 'D' CONFIG WITH 'VGG16' (USED ONLY TO COPY BENCHMARK)
        self.classifier = nn.Linear(math.ceil(self.params['HiddenLayerNumNodes'][-1]/self.params['MaxPoolArgs']['kernel_size']), self.num_classes)


        #weights initialization (this step can also be put below in a separate def)        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                #nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    #m.bias.detach().zero_()
                    nn.init.constant_(m.bias, 0)
        
        #self.weights_init() #call the orthogonal initial condition
        


    def forward(self, x):
        outs = {} 
        
        # flatten image input
        if  (self.params['Dataset']=='MNIST' ):
            x = x.view(-1,28*28)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (self.params['Dataset']=='CIFAR10' or self.params['Dataset']=='GaussBlob'):
            x = x.view(-1,32*32*3)
            
        L2 = self.features(x)
        outs['l2'] = L2
        Out = (self.classifier(L2))
        #Out = self.BenchClassifier(Out) #TODO: UNCOMMENT ABOVE LINE AND COMMENT THIS ONE (USED ONLY TO COPY BENCHMARK)
        outs['out'] = Out

        outs['pred'] = torch.argmax(Out, dim=1)
        
        #print(torch.sign(Fc2).int(), flush= True)
        #print(outs['pred'], flush= True)
        
        return outs

    

    def _make_layers(self):
        layers = []
        
        for index in range (len(self.params['HiddenLayerNumNodes'])-1):
            if index>0:
                layers += [nn.Linear(math.ceil(self.params['HiddenLayerNumNodes'][index]/self.params['MaxPoolArgs']['kernel_size']) , self.params['HiddenLayerNumNodes'][index+1])]
            else:
                layers += [nn.Linear(self.params['HiddenLayerNumNodes'][index], self.params['HiddenLayerNumNodes'][index+1])]
            if self.params['NetConf']=='RelU+MaxPool' : #setting self.params['MaxPoolArgs']['kernel_size'] we fall into the specific case of only ReLU
                if self.params['ShiftAF']=='ON':   
                    #layers+= [self.ShiftedReLU()]
                    layers+= [self.Null_Mean_ReLU]
                else:    
                    layers+= [nn.ReLU()]
                layers+= [nn.MaxPool1d(kernel_size=self.params['MaxPoolArgs']['kernel_size'])]
        
        #see * operator (Unpacking Argument Lists)
        #The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.
        #The syntax is to use the symbol * to take in a variable number of arguments; by convention, it is often used with the word args.
        #What *args allows you to do is take in more arguments than the number of formal arguments that you previously defined.
        return nn.Sequential(*layers) 



channels_seq = [16, 32, 32, 64, 32]


class Deep_CNN(nn.Module, NetVariables):
    def __init__(self, params):
        """
        Deep CNN architecture used to spot IGB
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code

        Returns
        -------
        None.

        """

        self.params = params.copy()
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class
        """
        super().__init__()
        super().__init__(self.params)
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        

        self.layer0 = self.Const_size_Conv_layers(channels_seq)

        if self.params['IGB_flag']=='ON':
            
            self.l1= [nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)]
            if self.params['BN_flag']=='B_AF':
                self.l1.append(nn.BatchNorm2d(16))
            self.l1.append(nn.ReLU())
            self.l1.append(nn.MaxPool2d(kernel_size=2, stride=2))
            if self.params['BN_flag']=='A_AF':
                self.l1.append(nn.BatchNorm2d(16))
            
            self.layer1 = nn.Sequential(*self.l1)
            
            
            self.l2= [nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)]
            if self.params['BN_flag']=='B_AF':
                self.l2.append(nn.BatchNorm2d(16))
            self.l2.append(nn.ReLU())
            self.l2.append(nn.MaxPool2d(kernel_size=4, stride=2))
            if self.params['BN_flag']=='A_AF':
                self.l2.append(nn.BatchNorm2d(16))            
            
            
            self.layer2 = nn.Sequential(*self.l2)
            
            
            
        #TODO: adapt to the structure above to add the batch norm
        elif self.params['IGB_flag']=='OFF':
            
            
            
            self.l1= [nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2)]
            if self.params['BN_flag']=='B_AF':
                self.l1.append(nn.BatchNorm2d(16))
            self.l1.append(nn.Tanh())
            self.l1.append(nn.AvgPool2d(kernel_size=2, stride=2))
            if self.params['BN_flag']=='A_AF':
                self.l1.append(nn.BatchNorm2d(16))
            
            self.layer1 = nn.Sequential(*self.l1)
            
            
            self.l2= [nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2)]
            if self.params['BN_flag']=='B_AF':
                self.l2.append(nn.BatchNorm2d(16))
            self.l2.append(nn.Tanh())
            self.l2.append(nn.AvgPool2d(kernel_size=4, stride=2))
            if self.params['BN_flag']=='A_AF':
                self.l2.append(nn.BatchNorm2d(16))            
            
            
            self.layer2 = nn.Sequential(*self.l2)
            
            
            #note the difference in values from the MNIST case in the following line; it is due to the different image size
            #in fact, a generic image (regardless of the number of channels) with size X*Y changes its extension in the 2 directions in the following way:
            #In the convolutional layer: X -> 1+(X-kernel_size + 2*padding)/stride
            # In the pooling layer: X -> 1+(X-kernel_size)/stride
        if  (self.params['Dataset']=='MNIST'):
            self.fc = nn.Linear(6*6*16, self.num_classes)
        elif  (self.params['Dataset']=='CIFAR10'):
            self.fc = nn.Linear(7*7*16, self.num_classes)                
            
        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition 
        
        
    def Const_size_Conv_layers(self, channels_seq):
        layers = []
        if  (self.params['Dataset']=='MNIST'):
            in_channels = 1
        elif  (self.params['Dataset']=='CIFAR10'): 
            in_channels = 3
        
        
        for x in channels_seq:
            if self.params['IGB_flag']=='OFF':
                
                self.l= [nn.Conv2d(in_channels, x, kernel_size=5, stride=1, padding=4)]
                if self.params['BN_flag']=='B_AF':
                    self.l.append(nn.BatchNorm2d(x))
                self.l.append(nn.Tanh())
                self.l.append(nn.AvgPool2d(kernel_size=5, stride=1))
                if self.params['BN_flag']=='A_AF':
                    self.l.append(nn.BatchNorm2d(x))      
                
                
                layers += self.l
                in_channels = x            

            elif self.params['IGB_flag']=='ON':
                
                self.l= [nn.Conv2d(in_channels, x, kernel_size=5, stride=1, padding=4)]
                if self.params['BN_flag']=='B_AF':
                    self.l.append(nn.BatchNorm2d(x))
                self.l.append(nn.ReLU())
                self.l.append(nn.MaxPool2d(kernel_size=5, stride=1))
                if self.params['BN_flag']=='A_AF':
                    self.l.append(nn.BatchNorm2d(x))            
                

                layers += self.l
                in_channels = x      
        #see * operator (Unpacking Argument Lists)
        #The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.
        #The syntax is to use the symbol * to take in a variable number of arguments; by convention, it is often used with the word args.
        #What *args allows you to do is take in more arguments than the number of formal arguments that you previously defined.
        return nn.Sequential(*layers) 

        
        
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
            

        
    def forward(self, x):
        outs = {}
        L0 = self.layer0(x)
        L1 = self.layer1(L0)
        outs['l1'] = L1
        L2 = self.layer2(L1)
        #After pooling or CNN convolution requires connection full connection layer, it is necessary to flatten the multi-dimensional tensor into a one-dimensional,
        #Tensor dimension convolution or after pooling of (batchsize, channels, x, y), where x.size (0) means batchsize value, and finally through x.view (x.size (0), -1) will be in order to convert the structure tensor (batchsize, channels * x * y), is about (channels, x, y) straightened, can then be connected and fc layer
        outs['l2'] = L2
        Out = L2.reshape(L2.size(0), -1)
        Out = self.fc(Out)
        outs['out'] = Out
        outs['pred'] = torch.argmax(Out, dim=1)
        return outs










class Net(nn.Module, NetVariables):
    def __init__(self, params):
        """
        Network class: this is a simple toy network  ()MLP with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below

        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code

        Returns
        -------
        None.

        """
        
        
        
        #super(Net,self).__init__()
        #super(Net, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)        
        
        
        # number of hidden nodes in each layer (512)
        hidden_1 = 32
        hidden_2 = 32

        # linear layer (784 -> hidden_1)
        if  (params['Dataset']=='MNIST'):
            self.fc1 = nn.Linear(28*28, hidden_1)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (params['Dataset']=='CIFAR10'):
            self.fc1 = nn.Linear(32*32*3, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1,hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, self.num_classes)
        
        #weights initialization (this step can also be put below in a separate def)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)
        #initialize the bias to 0
        nn.init.constant_(self.fc3.bias, 0)
    #I return from the forward a dictionary to get the output after each layer not only the last one
    #this is useful for example in the inter-classes angles (calculated throught the scalar product between inner layers representation)
    def forward(self,x):
        
        outs = {}
        # flatten image input
        if  (self.params['Dataset']=='MNIST'):
            x = x.view(-1,28*28)
        #NOTE: images in MNIST dataset are b&w photos with 28*28 pixels, CIFAR10 instead contain with colored photos (3 colour channel) with 32*32 pixels 
        elif  (self.params['Dataset']=='CIFAR10'):
            x = x.view(-1,32*32*3)
        # add hidden layer, with relu activation function
        Fc1 = F.relu(self.fc1(x))
        
        outs['l1'] = Fc1
        
        # add hidden layer, with relu activation function
        #Fc2 = F.relu(self.fc2(Fc1))
        #if you want to use a tanh activation function don't use F.tanh wich is deprecated; instead substitute the line above with the one below
        Fc2 = torch.tanh(self.fc2(Fc1))
        
        
        outs['l2'] = Fc2
        
        # add output layer
        Out = self.fc3(Fc2)
        
        outs['out'] = Out
        
        return outs


class ConvNet(nn.Module, NetVariables, OrthoInit):

    def __init__(self, params):
        """
        Network class: this is a prototipe of simple CNN with 2 hidden layer 
            In general to understand the architecture of the network is useful to read the forward method below
    
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code
    
        Returns
        -------
        None.
    
        """

        self.params = params.copy()
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class
        """
        super().__init__()
        super().__init__(self.params)
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        if  (self.params['Dataset']=='MNIST'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7*7*32, self.num_classes)
        elif  (self.params['Dataset']=='CIFAR10' or self.params['Dataset']=='INATURALIST' or self.params['Dataset']=='CIFAR100'):  
            
            self.l1=[nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)]
            if self.params['IGB_flag'] == 'ON':
                self.l1.append(nn.ReLU())
                self.l1.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif self.params['IGB_flag'] == 'OFF':    
                self.l1.append(nn.Tanh())
                self.l1.append(nn.AvgPool2d(kernel_size=2, stride=2))     
                
            self.layer1 = nn.Sequential(*self.l1)
            
            self.l2=[nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=2)]
            if self.params['IGB_flag'] == 'ON':
                self.l2.append(nn.ReLU())
                self.l2.append(nn.MaxPool2d(kernel_size=4, stride=4))
            elif self.params['IGB_flag'] == 'OFF':    
                self.l2.append(nn.Tanh())
                self.l2.append(nn.AvgPool2d(kernel_size=4, stride=4))     
                
            self.layer2 = nn.Sequential(*self.l2)
            


            #note the difference in values from the MNIST case in the following line; it is due to the different image size
            #in fact, a generic image (regardless of the number of channels) with size X*Y changes its extension in the 2 directions in the following way:
            #In the convolutional layer: X -> 1+(X-kernel_size + 2*padding)/stride
            # In the pooling layer: X -> 1+(X-kernel_size)/stride
            self.fc = nn.Linear(4*4*64, self.num_classes)                
            
        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
            

        
    def forward(self, x):
        outs = {}
        
        L1 = self.layer1(x)
        outs['l1'] = L1
        L2 = self.layer2(L1)
        #After pooling or CNN convolution requires connection full connection layer, it is necessary to flatten the multi-dimensional tensor into a one-dimensional,
        #Tensor dimension convolution or after pooling of (batchsize, channels, x, y), where x.size (0) means batchsize value, and finally through x.view (x.size (0), -1) will be in order to convert the structure tensor (batchsize, channels * x * y), is about (channels, x, y) straightened, can then be connected and fc layer
        outs['l2'] = L2
        Out = L2.reshape(L2.size(0), -1)
        Out = self.fc(Out)
        outs['out'] = Out
        outs['pred'] = torch.argmax(Out, dim=1)
        return outs




"""
class ConvNet(nn.Module, NetVariables, OrthoInit):

    def __init__(self, params):


        self.params = params.copy()
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class

        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        if  (self.params['Dataset']=='MNIST'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(7*7*32, self.num_classes)
        elif  (self.params['Dataset']=='CIFAR10'):  
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
                #nn.ReLU(),
                nn.Tanh(),
                #nn.BatchNorm2d(16), #PUT BATCH NORM TO CONFIRM THAT THIS IS THE PROBLEM FOR VGG16
                
                #nn.GroupNorm(int(16/self.params['group_factor']), 16),                
                
                #nn.MaxPool2d(kernel_size=2, stride=2)
                nn.AvgPool2d(kernel_size=2, stride=2)
                )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                #nn.ReLU(),
                nn.Tanh(),
                
                #nn.BatchNorm2d(32), #PUT BATCH NORM TO CONFIRM THAT THIS IS THE PROBLEM FOR VGG16
                
                #nn.GroupNorm(int(32/self.params['group_factor']), 32),
                
                #nn.MaxPool2d(kernel_size=2, stride=2)
                #use avgpool and tanh to prevent the guess imbalance
                nn.AvgPool2d(kernel_size=2, stride=2)
                )
            #note the difference in values from the MNIST case in the following line; it is due to the different image size
            #in fact, a generic image (regardless of the number of channels) with size X*Y changes its extension in the 2 directions in the following way:
            #In the convolutional layer: X -> 1+(X-kernel_size + 2*padding)/stride
            # In the pooling layer: X -> 1+(X-kernel_size)/stride
            self.fc = nn.Linear(8*8*32, self.num_classes)                
            
        self.initialize_weights() #calling the function below to initialize weights
        #self.weights_init() #call the orthogonal initial condition    
    def initialize_weights(self):
        #modules is the structure in which pytorch saves all the layers that make up the network
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            
            

        
    def forward(self, x):
        outs = {}
        
        L1 = self.layer1(x)
        outs['l1'] = L1
        L2 = self.layer2(L1)
        #After pooling or CNN convolution requires connection full connection layer, it is necessary to flatten the multi-dimensional tensor into a one-dimensional,
        #Tensor dimension convolution or after pooling of (batchsize, channels, x, y), where x.size (0) means batchsize value, and finally through x.view (x.size (0), -1) will be in order to convert the structure tensor (batchsize, channels * x * y), is about (channels, x, y) straightened, can then be connected and fc layer
        outs['l2'] = L2
        Out = L2.reshape(L2.size(0), -1)
        Out = self.fc(Out)
        outs['out'] = Out
        return outs

"""


#WARNING: VGG is designed only for CIFAR10 data    

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    
    #'VGG16': [64, 'D3', 64, 'M', 128, 'D4', 128, 'M', 256, 'D4', 256, 'D4', 256, 'M', 512,'D4', 512,'D4', 512, 'M', 512,'D4', 512,'D4', 512, 'M', 'D5'], #with dropout
    #'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,512, 'M'], #original
    #'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,512, 'A'], #original but with an averaging pool at the end instead of maxpool
    #'VGG16': [64, 'D3', 64, 'M', 128, 'D4', 128, 'M', 256, 'D4', 256, 'D4', 256, 'M', 512,'D4', 512,'D4', 512, 'M', 512,'D4', 512,'D4', 512, 'A', 'D5'], #original but with an averaging pool at the end instead of maxpool and with dropout
    'VGG16': [64, 'Dp', 64, 'M', 128, 'Dp', 128, 'M', 256, 'Dp', 256, 'Dp', 256, 'M', 512,'Dp', 512,'Dp', 512, 'M', 512,'Dp', 512,'Dp', 512, 'A', 'Dp'], #dropouts dependent from a single parameter (useful for hyper-par optim.) 
    'BENCHMARKVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module, NetVariables, OrthoInit): 
    #below lines modified to uniform classes input to CNN and MLP case (vgg_name is fixed to 'VGG16')
    #def __init__(self, vgg_name, n_out):
    def __init__(self, params):
        """
        Network class: this is a prototipe of more complex (deep) CNN: the sequence of layers can be chosen from one of the above dict item (cfg) 
            In general to understand the architecture of the network is useful to read the forward method below
        
        Note: all the module defined in Init method will be automatically charged on device (and so will be present on self.model.parameters); 
        This means that if you define modules in Init but don't use them in forward they will have Null grad (this will raise an error during the cloning of Grad) and waste useful memory for unused modules
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code
    
        Returns
        -------
        None.
    
        """
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class 
        self.params = params.copy()
        
        """        
        super(VGG, self).__init__()
        super(VGG, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        
        
        #self.features = self._make_layers(cfg[vgg_name])
        self.features = self._make_layers(cfg['VGG16']) #TODO:  SUBSTITUTE 'D' CONFIG WITH 'VGG16' (USED ONLY TO COPY BENCHMARK)
        self.classifier = nn.Linear(512, self.num_classes)
        """
        self.BenchClassifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        """

        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    #m.bias.detach().zero_()
                    nn.init.constant_(m.bias, 0)
        
        #self.weights_init() #call the orthogonal initial condition
        


    def forward(self, x):
        outs = {} 
        L2 = self.features(x)
        outs['l2'] = L2
        Out = L2.view(L2.size(0), -1)
        Out = self.classifier(Out)
        #Out = self.BenchClassifier(Out) #TODO: UNCOMMENT ABOVE LINE AND COMMENT THIS ONE (USED ONLY TO COPY BENCHMARK)
        outs['out'] = Out
        return outs

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif x=='A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                
            elif x == 'D3':
                layers += [nn.Dropout(0.3)]

            elif x == 'D4':
                layers += [nn.Dropout(0.4)]            

            elif x == 'D5':
                layers += [nn.Dropout(0.5)]   
                
            elif x == 'Dp':
                layers += [nn.Dropout(self.params['dropout_p'])] 
                
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.ReLU(inplace=True)
                           #nn.LeakyReLU(negative_slope=0.1, inplace=False),
                           nn.Tanh()  #put it back after banch runs
                           #,nn.BatchNorm2d(x)   #For now Batch Norm is excluded because it is incompatible with PCNGD, GD, PCNSGD where I forward sample by sample
                           ,nn.GroupNorm(int(x/self.params['group_factor']), x) #put it back after benchmark run
                           #,nn.GroupNorm(int(1), x)
                           ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        #see * operator (Unpacking Argument Lists)
        #The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.
        #The syntax is to use the symbol * to take in a variable number of arguments; by convention, it is often used with the word args.
        #What *args allows you to do is take in more arguments than the number of formal arguments that you previously defined.
        return nn.Sequential(*layers) 




class VGG_Custom_Dropout(nn.Module, NetVariables, OrthoInit):
    #below lines modified to uniform classes input to CNN and MLP case (vgg_name is fixed to 'VGG16')
    #def __init__(self, vgg_name, n_out):
    def __init__(self, params):

        """
        Network class: this is a prototipe of more complex (deep) CNN: the sequence of layers can be chosen from one of the above dict item (cfg) 
            In general to understand the architecture of the network is useful to read the forward method below
            The only difference with respect to the above VGG is that here you have control on the dropout layer defined by a mask.
            The right dropout is the one of class VGG (a different mask has to be applied on each image of the dataset (both for stochastic and deterministic algorithms)); this constitute an extension used for example to show that with conventional dropout PCNGD monotony prediction breaks (also for small lr)
    
        Parameters
        ----------
        params : dict
            dict of parameters imported from Main code
    
        Returns
        -------
        None.
    
        """        
        #this is an example of multiple inherance class
        #we need a single "super" call for each parent class 
        self.params = params.copy()
        
        """        
        super(VGG, self).__init__()
        super(VGG, self).__init__(self, params['n_out'], params['NSteps'], params['n_epochs'])
        """
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)
        
        
        
        #self.features = self._make_layers(cfg[vgg_name])
        self.ModuleDict = self._make_layers(cfg['VGG16'])
        self.classifier = nn.Linear(512, self.num_classes)
        self.mask_dict = {}


        """
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                #nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    #m.bias.detach().zero_()
                    nn.init.constant_(m.bias, 0)
        """
        self.weights_init() #call the orthogonal initial condition
        


    def forward(self, x, Mask_Flag):
        outs = {} 
        #iterate over the ModuleDict substituting where necessary identities appropriately (we use a flag to figure out when to update the masks)
        #flagghetta=0
        
        for key in self.ModuleDict:
            if key.startswith('!'):
                
                if not self.training: #if we are in eval mode the dropout is substitute by a identity layer
                    x = self.ModuleDict[key](x)
                else:
                
                
                    if Mask_Flag==1: #this flag trigger the update of masks 
                        self.mask_dict[key] = torch.distributions.Bernoulli(probs=(1-self.params['dropout_p'])).sample(x.size())
                        self.mask_dict[key] = self.mask_dict[key].to(self.params['device']) #load the mask used for the below tensor multiplication on the same device
                    """
                    if flagghetta==0:
                        print('x prima', x[0][0][0])
                        print('a maschera', self.mask_dict[key][0][0][0])                   
                    """

                    x = x * self.mask_dict[key] * 1/(1-self.params['dropout_p']) #dropout layer
                    
                    """
                    if flagghetta==0:                 
                        print('x dopo', x[0][0][0])
                    flagghetta=1                   
                    """
                    
            else: #for modules different from dropout regular forward
                x = self.ModuleDict[key](x)
        
        L2 = x
        outs['l2'] = L2
        Out = L2.view(L2.size(0), -1)
        Out = self.classifier(Out)
        outs['out'] = Out
        return outs

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        ModuleDict = nn.ModuleDict()
        NumberKey=0
        
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
            elif x=='A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                 
                
            elif x == 'Dp':
                
                ModuleDict[str(NumberKey)] = nn.Sequential(*copy.deepcopy(layers)) 
                NumberKey+=1
                ModuleDict['!'+str(NumberKey)] = nn.Identity()
                NumberKey+=1
                layers = []
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.ReLU(inplace=True),
                           #nn.LeakyReLU(negative_slope=0.1, inplace=False),
                           nn.Tanh()
                           #,nn.BatchNorm2d(x)   #For now Batch Norm is excluded because it is incompatible with PCNGD, GD, PCNSGD where I forward sample by sample
                           ,nn.GroupNorm(int(x/self.params['group_factor']), x)
                           #,nn.GroupNorm(int(1), x)
                           ]
                in_channels = x
        NumberKey+=1
        ModuleDict[str(NumberKey)] = nn.AvgPool2d(kernel_size=1, stride=1)
        #see * operator (Unpacking Argument Lists)
        #The special syntax *args in function definitions in python is used to pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.
        #The syntax is to use the symbol * to take in a variable number of arguments; by convention, it is often used with the word args.
        #What *args allows you to do is take in more arguments than the number of formal arguments that you previously defined.
        return ModuleDict




class PreTrainedResNet50(nn.Module, NetVariables, OrthoInit):
    def __init__(self, params, pretrained=True):
        
        self.params = params.copy()

        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)

        
        #super(PreTrainedResNet50, self).__init__()
        
        # Load a pre-trained ResNet50 model
        if pretrained:
            #self.weights = torchvision.models.ResNet50_Weights.DEFAULT
            
            #self.imagenet_transforms = self.weights.transforms()
            
            #if you need to use pretrained models in offline setting you have to first sore it in a dict and then directly import it
            self.pretrained_model = models.resnet50(pretrained=False)
            self.pretrained_model.load_state_dict(torch.load('./resnet50_weights.pth', map_location=torch.device(self.params['device'])))
            

            #self.pretrained_model = models.resnet50(pretrained=True)  #deprecated but if the torchvision version is old you have to use this line, otherwise the one below
            #self.pretrained_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        else:
            self.pretrained_model = models.resnet50(pretrained=False)
        
        # Modify the classifier head
        self.pretrained_model.fc = nn.Linear(self.pretrained_model.fc.in_features, self.num_classes)
        
        # Initialize the weights of the classifier head
        self._initialize_classifier_weights(self.pretrained_model.fc)

    def _initialize_classifier_weights(self, layer):
        """
        Initializes the weights of the classifier layer with Kaiming normal distribution
        and biases to zero.
        """
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        

        outs = {}
        
        outs['l2'] = torch.tensor([[0]]) #dummy item (for now we don't need the hidden representation of the signal)
        
        Out = self.pretrained_model(x)
        
        outs['out'] = Out

        outs['pred'] = torch.argmax(Out, dim=1)
        
        #print(torch.sign(Fc2).int(), flush= True)
        #print(outs['pred'], flush= True)
        
        return outs
        



class PreTrainedViT(nn.Module, NetVariables, OrthoInit):
    def __init__(self, params, model_name='vit_b_16', pretrained=True):

        self.params = params.copy()

        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)        
        
        
        # Load a pre-trained ViT model
        if pretrained:
            
            self.pretrained_model = models.vit_b_16(pretrained=False)
            self.pretrained_model.load_state_dict(torch.load('./vit_b_16_weights.pth', map_location=torch.device(self.params['device'])))
            
            # Use the torchvision.models.vit_* functions to load a pre-trained model
            #self.pretrained_model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            # Initialize without pre-trained weights
            self.pretrained_model = models.vit_b_16(weights=None)
        
        self.pretrained_model.heads.head = nn.Linear(self.pretrained_model.heads.head.in_features, self.num_classes)
        
        # Initialize the weights of the classifier head
        self._initialize_mlp_weights(self.pretrained_model.heads)

    def _initialize_mlp_weights(self, mlp_block):
        if isinstance(mlp_block, nn.Sequential):
            for layer in mlp_block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)
        elif isinstance(mlp_block, nn.Linear):
            nn.init.kaiming_normal_(mlp_block.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(mlp_block.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.
        """
        outs = {}
        
        # For ViT, we don't easily have access to intermediate representations before the head,
        # as it's a direct transformation from the last transformer block to the head.
        # Therefore, we keep this simple and direct for now.
        
        outs['l2'] = torch.tensor([[0]]) #dummy item (for now we don't need the hidden representation of the signal)

        
        Out = self.pretrained_model(x)
        
        outs['out'] = Out
        outs['pred'] = torch.argmax(Out, dim=1)
        
        return outs



class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network model.

    Args:
        input_dim (int): The number of input features.
        hidden_dim (int): The number of hidden units in each layer.
        output_dim (int): The number of output units.
        num_layers (int, optional): The number of layers in the MLP. Defaults to 5.

    Attributes:
        mlp (nn.Sequential): The sequential container for the MLP layers.

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5):
        super(MLP, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        return self.mlp(x)



class PreTrainedSwinT(nn.Module, NetVariables, OrthoInit):
    def __init__(self, params, pretrained=True):
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)        
        
        # Load a pre-trained Swin Transformer model
        if pretrained:
            

            self.pretrained_model = torchvision.models.swin_t(weights=None)
            self.pretrained_model.load_state_dict(torch.load('./swin_t_weights.pth', map_location=torch.device(self.params['device'])))
            """
            #if you have access to internet you can use the 2 lines below in place of the ones above
            weights = torchvision.models.Swin_T_Weights.DEFAULT
            self.pretrained_model = torchvision.models.swin_t(weights=weights)
            """
        else:
            self.pretrained_model = torchvision.models.swin_t(weights=None)
        

        # Adapt the Head of the model to match the number of target classes
        num_features = 768  # You can verify this by inspecting the model structure
        hidden_dim = 768  # Keeping the width constant
        num_classes = self.num_classes  # Number of target classes
        # Modify the classifier head to use a generic MLP
        self.pretrained_model.head = MLP(num_features, hidden_dim, num_classes, num_layers=5)
        
        # Initialize the weights of the classifier head
        self._initialize_mlp_weights(self.pretrained_model.head)

    def _initialize_mlp_weights(self, mlp):
        for layer in mlp.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    """
        # Modify the classifier head
        num_features = 768 # you can inspect the num_features defyining a dummy model, i.e. model = swin_t(weights=Swin_T_Weights.DEFAULT), with  print(model) (input dimension of the last layer, the classifier)
        self.pretrained_model.head = nn.Linear(num_features, self.num_classes) 
        
        # Initialize the weights of the classifier head
        self._initialize_mlp_weights(self.pretrained_model.head)

    def _initialize_mlp_weights(self, layer):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(layer.bias, 0)
    """
    
    def forward(self, x):
        """
        Forward pass of the model.
        """
        outs = {}
        out = self.pretrained_model(x)
        
        outs['out'] = out
        outs['pred'] = torch.argmax(out, dim=1)
        outs['l2'] = torch.tensor([[0]])  # Dummy item
        
        return outs


class PreTrainedEfficientNetV2(nn.Module, NetVariables, OrthoInit):
    def __init__(self, params, pretrained=True):
        self.params = params.copy()
        
        nn.Module.__init__(self)
        NetVariables.__init__(self, self.params)
        OrthoInit.__init__(self)        
        
    
        # Load a pre-trained EfficientNet V2-S model
        if pretrained:
            self.pretrained_model = torchvision.models.efficientnet_v2_s(pretrained=True)
            #self.pretrained_model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
        else:
            self.pretrained_model = torchvision.models.efficientnet_v2_s(pretrained=False)

        # Modify the classifier head to use an MLP
        num_features = self.pretrained_model.classifier[1].in_features
        hidden_dim = num_features  # Keeping the width constant
        num_classes = self.num_classes  # Assuming num_classes is provided in params
        self.pretrained_model.classifier[1] = MLP(num_features, hidden_dim, num_classes, num_layers=5)
        
        # Initialize the weights of the classifier head
        self._initialize_mlp_weights(self.pretrained_model.classifier[1])

    def _initialize_mlp_weights(self, mlp):
        for layer in mlp.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)


    """
        # Replace the classifier head with one matching the number of target classes
        num_features = self.pretrained_model.classifier[1].in_features
        self.pretrained_model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        # Initialize the weights of the new classifier head
        self._initialize_mlp_weights(self.pretrained_model.classifier[1])

    def _initialize_mlp_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
    """
    def forward(self, x):
        """
        Forward pass of the model.
        """
        outs = {}
        out = self.pretrained_model(x)
        
        outs['out'] = out
        outs['pred'] = torch.argmax(out, dim=1)
        outs['l2'] = torch.tensor([[0]])  # Dummy item
        
        return outs






#define here a simil-hinge loss function to quantify how distant are on average the points from the boundary of the classification
class Simil_HingeLoss(torch.nn.Module):

    def __init__(self):
        super(Simil_HingeLoss, self).__init__()

    def forward(self, output, target):

        hinge_loss = -torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0 #just selecting all the component of the tensor (is only one in this case) grater than 0  and putting them to 0
        return torch.sum(hinge_loss)




#we start creating our own dataset from the files created by "GaussianGenerator.py" each file contain a single tensor data; files are divided in folder according to their belonging class

class GaussBlobsDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        self.class_map = {}
        self.targets = []

        for family in os.listdir(data_root):
            family_folder = os.path.join(data_root, family)

            #create a mapping for the classes
            self.class_map[family] = int(family)

            for sample in os.listdir(family_folder):
                sample_filepath = os.path.join(family_folder, sample)
                self.samples.append((family, torch.load(sample_filepath)))
                
                self.targets.append(self.class_map[family]) 


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        class_name, data = self.samples[idx]
        class_id = self.class_map[class_name]
        
        #we don't need to convert label in tensor here; dataloader will do it for us (creating a batch tensor label)
        #class_id = torch.tensor([class_id])
        
        #return self.samples[idx]
        return data, class_id



#%% BLOCKS CLASS
class Bricks:
    """
    The following class has to be interpred as the bricks that will used to made up the main code 
    it contains all the blocks of code that performs simple tasks: the general structure structure is as follow:
        inside class Bricks we instantiate one of the Net classes 
        so in this case we will not use class inheritance but only class composition: 
            I don't use the previous class as super class but simply call them creating an istance inside the class itself; each of the Net classes inherit the class NetVariables (where important measures are stored)
            Notes for newcomers in python:
                
            Inheritance is used where a class wants to derive the nature of parent class and then modify or extend the functionality of it. 
            Inheritance will extend the functionality with extra features allows overriding of methods, but in the case of Composition, we can only use that class we can not modify or extend the functionality of it. It will not provide extra features.
            Warning: you cannot define a method that explicitly take as input one of the instance variables (variables defined in the class); it will not modify the variable value. 
            Instead if you perform a class composition as done for NetVariables you can give the variable there defined as input and effectively modify them           
                                
        NOTE: a part of the variable is defined in the class inhered from the network architecture class while a parts is definded in the following class itself.
        as a general rule the interesting variables are defined in the parent class, while the temp storing variables in the bricks of the following class
        THIS IS AN IMPORTANT POINT BECAUSE A COMMON MISTAKES WHEN DEALING WITH CLASSES IS TO REDIFINE VARIABLES SHELLINGTHE PARENT'S ONES
    """
    
    def __init__(self, params):
        #You are sharing a reference to a Python dictionary; use a copy if this was not intended; dict.copy() creates a shallow copy of a dictionary
        self.params = params.copy()
        """
        self.NetMode = params['NetMode']
        self.n_out = params['n_out']
        self.NSteps = params['NSteps']
        self.n_epochs = params['n_epochs']
        """
        # initialize the NN
        #create an istance of the object depending on NetMode
        if(self.params['NetMode']=='MultiPerceptron'):
            #self.model = Net(self.params['n_out'], self.params['NSteps'], self.params['n_epochs'])
            self.model = Net(self.params)
            
        elif(self.params['NetMode']=='CNN'):
            self.model = ConvNet(self.params)
            self.params['OutShape']='MultipleNodes' 
            
        elif(self.params['NetMode']=='VGG16'):
            self.model = VGG(self.params)
        elif(self.params['NetMode']=='VGG_Custom_Dropout'):
            self.model = VGG_Custom_Dropout(self.params)
            
        elif(self.params['NetMode']=='Single_HL'):
            #we define a dict to store the number of guesses for each sign (and how many null values)
            self.params['SignProp']={0:0, -1:0, 1:0}
            self.model = Single_HL(self.params)
            self.params['OutShape']='Single_Node'
            print("MODULE LIST")
            for module in self.model.named_modules():
                print(module)
        elif(self.params['NetMode']=='MC_Single_HL'):
            self.model = MC_Single_HL(self.params)
            self.params['OutShape']='MultipleNodes'
            print("MODULE LIST")
            for module in self.model.named_modules():
                print(module)
        elif(self.params['NetMode']=='Deep_HL'):
            self.model = Deep_HL(self.params)
            self.params['OutShape']='MultipleNodes' 
            print("MODULE LIST")
            for module in self.model.named_modules():
                print(module)
        elif(self.params['NetMode']=='Deep_CNN'):
            self.model = Deep_CNN(self.params)
            self.params['OutShape']='MultipleNodes' 
            
        elif(self.params['NetMode']=='PT_ResNet50'):
            self.model = PreTrainedResNet50(self.params)
            self.params['OutShape']='MultipleNodes' 
        elif(self.params['NetMode']=='PT_ViT'):
            self.model = PreTrainedViT(self.params)
            self.params['OutShape']='MultipleNodes' 
        elif(self.params['NetMode']=='PT_SwinT'):
            self.model = PreTrainedSwinT(self.params)
            self.params['OutShape']='MultipleNodes'  
        elif(self.params['NetMode']=='PT_EN2'):
            self.model = PreTrainedEfficientNetV2(self.params)
            self.params['OutShape']='MultipleNodes'  
        else:
            print('Architecture argument is wrong', file = self.params['WarningFile'])
        self.RoundSolveConst = 1e3 #NOTE: this is a scaling factor to prevent underflow problem in norm computation/dot product if you have a large vector with small components; If this is not the case remove it to avoid the opposite problem (overflow)
        self.Epsilon = 1e-6
        
        self.NormGrad1 = [[] for i in range(self.model.num_classes)] 
        self.NormGrad2 = [[] for i in range(self.model.num_classes)] 
        #self.NormGradOverlap =  np.zeros((self.model.num_classes, self.params['samples_grad']))
        self.NormGrad1Tot = []
        self.NormGrad2Tot = []
        self.cos_alpha = 0 #initialize the angle for the scheduling
        
        self.model.double() #Call .double() on the model and input, which will transform all parameters to float64:
    



    #prepare the dataset and load it on the device
    def DataLoad(self):
        """
        transform dataset (convert to tensor and standardize) and wrap it in dataloader.
        If specified a selection of class and a reduction of dataset is done, (the reduction can be different for the classes if class imbalance is triggered)
        to parallelize the computation (not forward sample by sample) here we define a dataloader for each class ; in this way:
            we can define, for each class a batch with only element of the same class (and so be able to forward the whole batch and computing the corresponding per class gradient)
            we can define very big batch since we forward every time sub-batches defined on the single classes
        Returns
        -------
        None.

        """
        
            
        # convert data to torch.FloatTensor
        #transform = transforms.ToTensor()
        
        #convert data to tensor and standardize them (rescale each channel of each image fixing the mean to 0 and the std to 1)
        
        #here we compute mean and std to standardize the dataset; we repeat the procedure every time since we could be working with a subset of classes
        DataMean = DatasetMeanStd(self.params).Mean()
        DataStd = DatasetMeanStd(self.params).Std()
        print("the Mean and Std used to standardize data are {} and {}".format(DataMean, DataStd))
        
        #TODO: correct standardization for the mnist dataset below
        if (self.params['Dataset']=='MNIST'):
            self.transform = transforms.Compose([
                        transforms.ToTensor() #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                        ,transforms.Normalize(mean=(DataMean,), std=(DataStd,))
                        ])   
        elif(self.params['Dataset']=='CIFAR10') or (self.params['Dataset']=='CIFAR100'):    
            if self.params['Pre_Trained_Flag']=='ON':
                """
                self.transform = transforms.Compose([
                        transforms.Resize(224),
                        self.imagenet_transforms]) #in this way we standardize only on the subset of dataset used
                        #transforms.Normalize((0.49236655, 0.47394478, 0.41979155), (0.24703233, 0.24348505, 0.26158768))]) 
                """
                # Define transformations: resize, crop, convert to tensor, and normalize
                # These normalization values are standard for models trained on ImageNet
                self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                
            
            else:
                if self.params['Resize_Flag']=='ON':
                    self.transform = transforms.Compose([
                            transforms.Resize(224),
                            transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                            #NOTE: the standardization depend on the dataset that you use; if you use a subset of classes you have to calculate mean and std on the restricted dataset
                            transforms.Normalize(DataMean, DataStd)]) #in this way we standardize only on the subset of dataset used
                            #transforms.Normalize((0.49236655, 0.47394478, 0.41979155), (0.24703233, 0.24348505, 0.26158768))]) 
                               
                else:
                    self.transform = transforms.Compose([
                            transforms.ToTensor(), #NOTE: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                            #NOTE: the standardization depend on the dataset that you use; if you use a subset of classes you have to calculate mean and std on the restricted dataset
                            transforms.Normalize(DataMean, DataStd)]) #in this way we standardize only on the subset of dataset used
                            #transforms.Normalize((0.49236655, 0.47394478, 0.41979155), (0.24703233, 0.24348505, 0.26158768))]) 
                
        #to check the above values used for the dataset standardization you can use the function Mean and Std from DatasetMeanStd class (CodeBlocks module); below an example for the mean
        """
        a = []
        a = CodeBlocks.DatasetMeanStd('CIFAR10').Mean()/
        print(a)
        """
        
        # choose the training and testing datasets
        
        if (self.params['Dataset']=='MNIST'):
            print('this run used MNIST dataset', file = self.params['info_file_object'])
            self.train_data = datasets.MNIST(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.MNIST(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
            self.valid_data = datasets.MNIST(root = self.params['DataFolder'], train = False, download = True, transform = self.transform)
            num_data = len(self.train_data)
            i=0
            for item in self.train_data:
                if i==0:
                    print(item)
                    print('image', item[0].size())
                    print('label', item[1])
                    print('image', item[0])
                    i+=1
            print("total number of samples ", num_data, self.train_data.data.size())
        elif(self.params['Dataset']=='CIFAR10'):
            print('this run used CIFAR10 dataset', file = self.params['info_file_object'])
            self.train_data = datasets.CIFAR10(root = self.params['DataFolder'], train = True, download = False, transform = self.transform)
            self.test_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = False, transform = self.transform) 
            self.valid_data = datasets.CIFAR10(root = self.params['DataFolder'], train = False, download = False, transform = self.transform) 
            num_data = len(self.train_data)
            i=0
            for item in self.train_data:
                if i==0:
                    print(item)
                    print('image', item[0].size())
                    print('label', item[1])
                    print('image', item[0])
                    i+=1
            #print("total number of samples ", self.train_data.data)
            
        elif(self.params['Dataset']=='CIFAR100'):
            print('this run used CIFAR100 dataset', file = self.params['info_file_object'])
            self.train_data = datasets.CIFAR100(root = self.params['DataFolder'], train = True, download = True, transform = self.transform)
            self.test_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
            self.valid_data = datasets.CIFAR100(root = self.params['DataFolder'], train = False, download = True, transform = self.transform) 
            num_data = len(self.train_data)
            
        elif(self.params['Dataset']=='GaussBlob'):
            #TODO: WARNING you are using same dataset for train test valid (because for now you only need the train) for the GaussBlob dataset. You will need to modify this later
            self.train_data = GaussBlobsDataset(self.params['DataFolder'])
            self.test_data = GaussBlobsDataset(self.params['DataFolder'])
            self.valid_data = GaussBlobsDataset(self.params['DataFolder'])
            num_data = len(self.train_data)
            
            print("total number of samples ", num_data, self.train_data.data.size())
        else:
            print('the third argument you passed to the python code is not valid', file = self.params['WarningFile'])
        
        
        
        
        
        #A CLEANER APPROACH TO SELECT THE CLASSES
        if((self.params['ClassSelectionMode']=='ON') and (self.params['ClassImbalance'] == 'OFF')):
            #we start creating a map (throught a dictionary {realLabel: newLabel} ) betweeen the classes that we want to select and a ordered list of label
            
            self.label_map = self.params['label_map']
            #NOTE: when you modify the above dict you have tomodify also the classes appeearing in the following 2 conditions accordingly
            self.train_data = [(img, self.label_map[label]) for img, label in self.train_data if label in [1,9]]
            self.test_data = [(img, self.label_map[label]) for img, label in self.test_data if label in [1,9]]
            

            print('this run used this subset of classes ({original dataset labels : new mapped label for the simulation}): ', self.label_map, file = self.params['info_file_object'])
        elif(self.params['ClassSelectionMode']=='OFF'):
            print('this run used all the classes of the dataset:', file = self.params['info_file_object'])
            #num_classes=10
            
            
        #DATASET SELECTION FOR UNBALANCED CASE
        #define a variable to fix the unbalance rate between the 2 classes
        if((self.params['ClassSelectionMode']=='ON') and (self.params['ClassImbalance'] == 'ON')):
            
            #WARNING: THE MNIST CASE SHOW AN ERROR USING MULTIPLE DATALOADER, THEREFORE FOR NOW WE WILL DEFINE A SINGLE ONE
            if (self.params['Dataset']=='MNIST'):
                self.TrainDL = {}#dict to store data loader (one for each mapped class) for train set
                self.TestDL = {}#dict to store data loader (one for each mapped class) for test set
                self.ValidDL = {}#dict to store data loader (one for each mapped class) for valid set                
                
            else:
                self.TrainDL = {}#dict to store data loader (one for each mapped class) for train set
                self.TestDL = {}#dict to store data loader (one for each mapped class) for test set
                self.ValidDL = {}#dict to store data loader (one for each mapped class) for valid set
                #define the batch sizr for each class such that their proportion will be near to "self.params['ImabalnceProportions']"
                #the advantage of proceding like that is that we can easly get the exact same number of batches per each class
                if self.params['OversamplingMode'] == 'OFF':
                    #the batch size for each input class
                    self.TrainClassBS = np.rint((self.params['batch_size']/np.sum(self.params['ImabalnceProportions']))*np.divide(self.params['ImabalnceProportions'], self.params['MappedClassOcc'])).astype(int)
                    #the batch size for the whole associated output class given simply by the above expession multiplyed for the occurrences of each output class in the mapping
                    self.TrainTotalClassBS = (np.rint((self.params['batch_size']/np.sum(self.params['ImabalnceProportions']))*np.divide(self.params['ImabalnceProportions'], self.params['MappedClassOcc'])).astype(int)*(self.params['MappedClassOcc'])).astype(int)
                elif self.params['OversamplingMode'] == 'ON':
                    self.TrainClassBS = np.rint((self.params['batch_size']/self.model.num_classes)*np.reciprocal(self.params['MappedClassOcc'])).astype(int)
                    self.TrainTotalClassBS = ((np.rint((self.params['batch_size']/self.model.num_classes)*np.reciprocal(self.params['MappedClassOcc'])).astype(int))*(self.params['MappedClassOcc'])).astype(int)
                print("real size of the batch size of the training set (after the roundings): {}".format(np.sum(self.TrainTotalClassBS)),flush=True, file = self.params['info_file_object']) 
                print("the total sizes of mapped classes are {}".format(self.TrainTotalClassBS))
                
                MajorInputClassBS = np.amax(self.TrainClassBS) #we select here the class with greater element in the batch; that one will establish the bottle neck for the dataset, we assign to it the maximum possible number of element            

            self.traintargets = torch.tensor(self.train_data.targets) #convert label in tensors
            self.validtargets = torch.tensor(self.valid_data.targets) #convert label in tensors
            self.testtargets = torch.tensor(self.test_data.targets) #convert label in tensors
            #first we cast the target label (originary a list) into a torch tensor
            #we then define a copy of them to avoid issue during the class mapping 
                #can happen for example, using only the above one that I want to map {0:1, 1:0} 
                #we map the 0s in 1 and then map 1s to 0 the list of 1s will include also the 0s mapped in the precedent steps; to avoid so we follow the following rule:
            self.train_data.targets = torch.tensor(self.train_data.targets)
            self.valid_data.targets = torch.tensor(self.valid_data.targets)
            self.test_data.targets = torch.tensor(self.test_data.targets)
            
            if (self.params['Dataset']=='MNIST'):
                for key in self.params['label_map']:
                    self.trainTarget_idx = (self.traintargets==key).nonzero() 
                    self.validTarget_idx = (self.validtargets==key).nonzero()
                    if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                        self.testTarget_idx = (self.testtargets==key).nonzero()
              
                    
            elif (self.params['Dataset']!='MNIST'):
            
                self.TrainIdx = {}
                self.ValidIdx = {}
                self.TestIdx = {}
                for key in self.params['label_map']:
                    print("the batch size for the class {}, mapped in {} is {}".format(key, self.params['label_map'][key], self.TrainClassBS[self.params['label_map'][key]]),flush=True, file = self.params['info_file_object'])
                    #we start collecting the index associated to the output classes togheter
                    #TRAIN
                    self.trainTarget_idx = (self.traintargets==key).nonzero() 
                    #l0=int(900/MajorInputClassBS)*self.TrainClassBS[self.params['label_map'][key]] #just for debug purpose
                    l0 = int(len(self.trainTarget_idx)/MajorInputClassBS)*self.TrainClassBS[self.params['label_map'][key]] #we first compute the numbers of batches for the majority class and then reproduce for all the others in such a way they will have same number of batches but with a proportion set by self.TrainClassBS[classcounter-1]            
                    #self.Trainl0 = l0
                    #WARNING: LINE ABOVE CHANGED WITH THE ONE BELOW ONLY FOR DEBUG PURPOSE (MNIST ADAPTATION) RESUBSTITUTE WITH THE ONE ABOVE ONCE SOLVED THE ISSUE
                    self.Trainl0 = 500
                    print("the number of elements selected by the class {} loaded on the trainset is {}".format(key, self.Trainl0),flush=True, file = self.params['info_file_object'])
                    #print(self.trainTarget_idx)
                    ClassTempVar = '%s'%self.params['label_map'][key]
                    
                    #VALID
                    self.validTarget_idx = (self.validtargets==key).nonzero()
                    self.Validl0= 150 #should be less than 500 (since the total test set has 1000 images per class)                
                    #TEST
                    if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                        self.testTarget_idx = (self.testtargets==key).nonzero()
                        self.Testl0= 150 #should be less than 500 (since the total test set has 1000 images per class)
                    
                    if ClassTempVar in self.TrainIdx: #if the mapped class has already appeared, we concatenate the new indeces to the existing ones
                        self.TrainIdx['%s'%self.params['label_map'][key]] = torch.cat((self.TrainIdx['%s'%self.params['label_map'][key]], self.trainTarget_idx[:][0:self.Trainl0]),0)
                        self.ValidIdx['%s'%self.params['label_map'][key]] = torch.cat((self.ValidIdx['%s'%self.params['label_map'][key]], self.validTarget_idx[:][0:self.Validl0]),0)
                        if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                            self.TestIdx['%s'%self.params['label_map'][key]] = torch.cat((self.TestIdx['%s'%self.params['label_map'][key]], self.testTarget_idx[:][-self.Testl0:]),0)                   
                    else: #if, instead the class is selected for the first time, we simply charge it on the indeces dict
                        self.TrainIdx['%s'%self.params['label_map'][key]] = self.trainTarget_idx[:][0:self.Trainl0]
                        self.ValidIdx['%s'%self.params['label_map'][key]] = self.validTarget_idx[:][0:self.Validl0] #select the last indeces for the validation so we don't have overlap increasing the size
                        if (self.params['ValidMode']=='Test'): #if we are in testing mode we have to repeat it for a third dataset
                            self.TestIdx['%s'%self.params['label_map'][key]] = self.testTarget_idx[:][-self.Testl0:] #select the last indeces for the validation so we don't have overlap increasing the size
                
            #REMAP THE LABELS: now that the indexes are fixed we map the dataset to the new labels
            for key in self.params['label_map']:               
                self.train_data.targets[self.traintargets==key]= self.params['label_map'][key] 
                self.valid_data.targets[self.validtargets==key]=self.params['label_map'][key]
                if (self.params['ValidMode']=='Test'):
                    self.test_data.targets[self.testtargets==key]=self.params['label_map'][key]
            #print(self.train_data.targets)    
                    
            #DATALOADER CREATION    
            if (self.params['Dataset']=='MNIST'):
                    self.TrainDL['Class0'] = torch.utils.data.DataLoader(self.train_data, batch_size = self.params['batch_size'], 
                                                           shuffle=True , num_workers = self.params['num_workers'])                
                    self.ValidDL['Class0'] = torch.utils.data.DataLoader(self.valid_data, batch_size = self.params['batch_size'], #note that for test and valid the choice of the batch size is not relevant (we use these dataset only in eval mode)
                                                           shuffle=True , num_workers = self.params['num_workers']) 

                    if (self.params['ValidMode']=='Test'):                    
                        self.TestDL['Class0'] = torch.utils.data.DataLoader(self.test_data, batch_size = self.params['batch_size'], 
                                                           shuffle=True , num_workers = self.params['num_workers'])     
                                         
            else:
                #now we iterate over the mapped classes avoiding repetition
                print("checking that the data loader corresponding to each class contains the same number of batches",flush=True, file = self.params['info_file_object'])
                for MC in set(list(self.params['label_map'].values())): #with this syntax we avoid repetition of same dict items
                    #TRAIN
                    self.train_sampler = SubsetRandomSampler(self.TrainIdx['%s'%MC])  
                    #if we are studing the class imbalance case we use the sampler option to select data
                    #we load the dataloader corresponding to the mapped "self.params['label_map'][key]" class as a dict element
                    self.TrainDL['Class%s'%MC] = torch.utils.data.DataLoader(self.train_data, batch_size = self.TrainTotalClassBS[MC].item(), 
                                                           sampler = self.train_sampler, num_workers = self.params['num_workers'])     
    
                    #VALID
                    self.valid_sampler = SubsetRandomSampler(self.ValidIdx['%s'%MC])
                    self.ValidDL['Class%s'%MC] = torch.utils.data.DataLoader(self.valid_data, batch_size = self.params['batch_size'], #note that for test and valid the choice of the batch size is not relevant (we use these dataset only in eval mode)
                                                           sampler = self.valid_sampler, num_workers = self.params['num_workers']) 
                    
                    if (self.params['ValidMode']=='Test'):                    
                        self.test_sampler = SubsetRandomSampler(self.TestIdx['%s'%MC])                    
                        self.TestDL['Class%s'%MC] = torch.utils.data.DataLoader(self.test_data, batch_size = self.params['batch_size'], 
                                                               sampler = self.test_sampler, num_workers = self.params['num_workers'])     
            
                    #check that the data loader corresponding to each class contain the same number of batches;
    
                    print("the classes mapped in {} contains in its training dataloader {} batches".format(MC , len(self.TrainDL['Class%s'%MC])),flush=True, file = self.params['info_file_object']) 
                
                      
                
            """
            #WARNING: CAHNGED ABOVE BLOCK WITH THE FOLLOWING ONE
            #DATALOADER CREATION    
            #now we iterate over the mapped classes avoiding repetition
            print("checking that the data loader corresponding to each class contain the same number of batches",flush=True, file = self.params['info_file_object'])
            for MC in set(list(self.params['label_map'].values())): #with this syntax we avoid repetition of same dict items
                #TRAIN
                self.SubsetData = torch.utils.data.Subset(self.train_data, self.TrainIdx['%s'%MC])  
                #if we are studing the class imbalance case we use the sampler option to select data
                #we load the dataloader corresponding to the mapped "self.params['label_map'][key]" class as a dict element
                self.TrainDL['Class%s'%MC] = torch.utils.data.DataLoader(self.SubsetData, batch_size = self.TrainTotalClassBS[MC].item(), 
                                                       shuffle=True, num_workers = self.params['num_workers'])     

                #VALID
                self.SubsetData = torch.utils.data.Subset(self.valid_data,self.ValidIdx['%s'%MC])
                self.ValidDL['Class%s'%MC] = torch.utils.data.DataLoader(self.SubsetData, batch_size = self.params['batch_size'], #note that for test and valid the choice of the batch size is not relevant (we use these dataset only in eval mode)
                                                       shuffle=True, num_workers = self.params['num_workers']) 
                
                if (self.params['ValidMode']=='Test'):                    
                    self.SubsetData = torch.utils.data.Subset(self.test_data,self.TestIdx['%s'%MC])                    
                    self.TestDL['Class%s'%MC] = torch.utils.data.DataLoader(self.SubsetData, batch_size = self.params['batch_size'], 
                                                           shuffle=True, num_workers = self.params['num_workers'])     
        
                #check that the data loader corresponding to each class contain the same number of batches;

                print("the classes mapped in {} contains in its training dataloader {} batches".format(MC , len(self.TrainDL['Class%s'%MC])),flush=True, file = self.params['info_file_object']) 
            """

        next(iter(self.TrainDL['Class0']))        
        print('success!')        

        if(self.params['ClassImbalance'] == 'OFF'):
            self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size = self.params['batch_size'],
                                                     num_workers = self.params['num_workers'])
            
            self.valid_loader = torch.utils.data.DataLoader(self.test_data, batch_size = self.params['batch_size'],
                                                     num_workers = self.params['num_workers'])
                 
        #identify the number of classes from the number of outcome (in the label vector of the training set) with frequency non-zero

        self.SamplesClass = np.zeros(self.params['n_out'])
        self.TestSamplesClass = np.zeros(self.params['n_out'])
        self.ValidSamplesClass = np.zeros(self.params['n_out'])
        ind=0
        self.TrainTotal = 0
        
#         for data, label in self.valid_loader:
#             for im in range(0, len(label)):
#                 self.TestSamplesClass[label[im]] +=1
        if (self.params['Dataset']=='MNIST'):
            train_classes = [x for x in self.train_data.targets ]
            Train_number_classes = Counter(i.item() for i in train_classes)
            #print('elements for each class: ', Train_number_classes, train_classes)
            for key in Train_number_classes:
                self.SamplesClass[key] = Train_number_classes[key]
                self.TrainTotal += self.SamplesClass[key]
            
                print("train  number of samples in  {} is {}".format(key, self.SamplesClass[key]), flush = True, file = self.params['info_file_object'])
            print("total train  number of samples is {}".format( self.TrainTotal), flush = True, file = self.params['info_file_object'])
            self.ValTotal = 0
            valid_classes = [x for x in self.valid_data.targets]
            Valid_number_classes = Counter(i.item() for i in valid_classes) 
            print(Valid_number_classes)
            for key in Valid_number_classes:
                self.ValidSamplesClass[key] = Valid_number_classes[key]
                self.ValTotal += self.ValidSamplesClass[key]
            
                print("train  number of samples in  {} is {}".format(key, self.ValidSamplesClass[key]), flush = True, file = self.params['info_file_object'])
            print("total train  number of samples is {}".format( self.ValTotal), flush = True, file = self.params['info_file_object'])
    
            if (self.params['ValidMode']=='Test'):  
                self.TestTotal = 0
                test_classes = [x for x in self.test_data.targets]
                Test_number_classes = Counter(i.item() for i in test_classes) 
                print(Test_number_classes)
                for key in Test_number_classes:
                    self.TestSamplesClass[key] = Test_number_classes[key]
                    self.TestTotal += self.TestSamplesClass[key]
                
                    print("train  number of samples in  {} is {}".format(key, self.TestSamplesClass[key]), flush = True, file = self.params['info_file_object'])
                print("total train  number of samples is {}".format( self.TestTotal), flush = True, file = self.params['info_file_object'])
    
 
        else:
        
            for key in self.TrainDL:
                self.SamplesClass[ind] = len(self.TrainDL[key].sampler) #nota che con len(self.TrainDL[key]) ottieni invece il numero di batches
                self.TrainTotal += self.SamplesClass[ind]
                print("train  number of samples in  {} is {}".format(key, self.SamplesClass[ind]), flush = True, file = self.params['info_file_object'])
                ind+=1
            print("total train  number of samples is {}".format( self.TrainTotal), flush = True, file = self.params['info_file_object'])
            ind=0
            self.TestTotal = 0
            for key in self.TestDL:
                self.TestSamplesClass[ind] = len(self.TestDL[key].sampler)
                
                self.TestTotal += self.TestSamplesClass[ind]
                print("test number of samples in  {} are {}".format(key, self.TestSamplesClass[ind]), flush = True, file = self.params['info_file_object'])
                ind+=1            
            ind=0
            self.ValTotal = 0
            for key in self.ValidDL:
                self.ValidSamplesClass[ind] = len(self.ValidDL[key].sampler)
                self.ValTotal += self.ValidSamplesClass[ind]
                print("valid number of samples in  {} are {}".format(key, self.ValidSamplesClass[ind]), flush = True, file = self.params['info_file_object'])
                ind+=1
                
                    


    

        
    #load the model on the device and define the loss function and the optimizer object
    def NetLoad(self):      
        """
        This method load the Net on the specified device and specify the criterion (loss) and optimized to use

        Returns
        -------
        None.

        """
        #load model into the device
        self.model.to(self.params['device'])
        
        
        """
        # Pytorch will only use one GPU by default.  we need to make a model instance and check if we have multiple GPUs. If we have multiple GPUs, we can wrap our model using nn.DataParallel. Then we can put our model on GPUs by model.to(device)
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          model = nn.DataParallel(model)
        """
                
        # specify loss function (categorical cross-entropy)
        #NOTE: reduction mode
        #l'output layer (che va confrontato con i label reali) ha, in generale una forma (BS, Nc, d1,...,dk), dove BS indica il numero di immagini nella batch, Nc il numero di classi,  d1,...,dk ulteriori eventuali dimensioni
        #reduction='mean' raggruppa tutte le loss della batch sommandole e le normalizza per il numero di elementi*eventuali dimensioni (BS*d1*..*dk)
        #reduction='sum' effettua semplicemente la somma delle loss associate ad ogni elemento della batch
        self.criterion = nn.CrossEntropyLoss(reduction='sum')#reduction='sum'
        self.SampleCriterion = nn.CrossEntropyLoss(reduction = 'none')
        self.mean_criterion = nn.CrossEntropyLoss(reduction='mean')
        
        #a different loss function that can be used for example in case of single output 
        self.MSELoss = nn.MSELoss(reduction='sum')
        self.Simil_HingeLoss = Simil_HingeLoss()
        
        
        
        # specify optimizer (stochastic gradient descent) and learning rate
        #To use torch.optim you have to construct an optimizer object, that will hold the current state and will update the parameters based on the computed gradients.
        #To construct an Optimizer you have to give it an iterable containing the parameters (all should be Variable s) to optimize. Then, you can specify optimizer-specific options such as the learning rate, weight decay, etc.
        #why not declare it in the init method?
        #If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it. Parameters of a model after .cuda() will be different objects with those before the call.
        #In general, you should make sure that optimized parameters live in consistent locations when optimizers are constructed and used.
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr = self.params['learning_rate'], momentum = self.params['momentum'], weight_decay=self.params['weight_decay'])
        
    def DefineRetrieveVariables(self):
        """
        creation of the temp variables; it initialize a list of empty list (one for each class) that usually are initialized at the beginning of the run (see InitialState method)

        Returns
        -------
        None.

        """
        self.MeanRepresClass = [[] for i in range(self.params['n_out'])]
        self.MRC = [[] for i in range(self.params['n_out'])]       
        self.EvaluationVariablesReset() #reset of temp evaluation variables
        self.StoringGradVariablesReset()
    
    def InitialState(self):
        """
        Iterate over the train loader calculating the mean representation per class (MRC), the per class loss 
        and the representation norm at the beginning of the train to evaluate the initial state of the system
        This version i adapted for the case with dataset divided in classes

        Returns
        -------
        None.

        """        
        
        #Starting measure to see the initial state (without any training)
        self.model.eval() 
        #creation of the temp variables; i initialize a list of empty list (one for each class)
        self.MeanRepresClass = [[] for i in range(self.params['n_out'])]
        self.MRC = [[] for i in range(self.params['n_out'])]

        self.EvaluationVariablesReset() #reset of temp evaluation variables
        #we evaluate the training set at the times before training starts
        self.StoringGradVariablesReset()
        
        for EvalKey in self.TrainDL:
            #print(EvalKey)
            SetFlag = 'Train' 
            #print('fine', self.TrainDL[EvalKey][0], flush=True)
            for dataval,labelval in self.TrainDL[EvalKey]:
        
                Mask_Flag = 1
                
                dataval = dataval.double() 
                dataval = dataval.to(self.params['device'])
                labelval = labelval.to(self.params['device']) 

                if self.params['NetMode']=='VGG_Custom_Dropout':
                    
                    self.DropoutBatchForward(dataval, Mask_Flag)
                    Mask_Flag = 0
                else:
                    self.BatchForward(dataval)
                    
                    
                self.output = self.OutDict['out'].clone()
                last_layer_repr = self.OutDict['l2'].clone()  
                    
                self.BatchEvalLossComputation(labelval, 0, SetFlag) #computation of the loss function and the gradient (with backward call)

                #computation of quantity useful for precision accuracy,... measures
                self.CorrectBatchGuesses(labelval, 0, SetFlag)
                #Store the last layer mean representation and per classes loss function
                """
                #ADAPT COMP. OF THE LAST LAYER TO THE CASE OF BATCHES (also remember the layer compression line).
                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                else:
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                """

                self.loss.backward()   # backward pass: compute gradient of the loss with respect to model parameters
                self.GradCopyUpdate(labelval[0]) #again we select one of the label scalar; since all the element in the batch represent the same class they are all equivalent
                self.optimizer.zero_grad()
                        
                    
                    

 
                    
            #putting gradient to 0 before filling it with the per class normalized sum
            self.optimizer.zero_grad()
            """
            NetInstance.LastLayerRepresCompression()
            """
            
            
        self.LossAccAppend(0, SetFlag)
            
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses

            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            
            if((self.params['Dynamic']=='PCNGD') or (self.params['Dynamic']=='GD')):
                self.model.StepGradientClassNorm[index][0] = self.Norm[index]
            else:
                self.model.StepGradientClassNorm[index][0] = self.Norm[index]/len(self.TrainDL['Class0']) #normalize for the number of batches in the dataset since I am computing the gradient norm relative to a single step (note that I use the number of batches of the majority class (always mapped to 0))

        
        
        
        """
        #you can write the following iteration in more compact way:
        #from itertools import combinations
        #for i, j in combinations(range(N), 2):
        AngleIndex=0
        for i,j in((i,j) for i in range(self.params['n_out']) for j in range(i)):
            self.model.TrainAngles[AngleIndex][0] = math.acos(np.sum(np.multiply(self.MRC[i].cpu().detach().numpy(), self.MRC[j].cpu().detach().numpy()))/(self.model.RepresentationClassesNorm[i][0]*self.model.RepresentationClassesNorm[j][0]))
            AngleIndex+=1 
        """

        #gradient angles computation
        AngleIndex=0
        print('num_classes', self.model.num_classes)
        print(len(self.GradCopy))
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
            print('indici', i, j)
            print(len(self.GradCopy[i]), len(self.GradCopy[j]))
            
            #calculate the cos(ang) as the scalar product normalized with the l2 norm of the vectors
            ScalProd=0

            TGComp=0
            for obj in self.GradCopy[i]:
                ScalProd += np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst))


                TGComp +=1
            ScalProd = ScalProd/(self.RoundSolveConst*self.RoundSolveConst)       
            
            #saving angle between 2 classes in a variable used for the lr_rate schedule
            #TODO: the following works only for the 2 classes problem (only one angle) pay attention in passing to more classes
            self.cos_alpha = ScalProd/(self.Norm[i]*self.Norm[j])
            
            print("The Scalar product between class {} and {} and corresponding norms are: {} {} {}, il prodotto delle 2 norme {}".format(i,j,ScalProd, self.Norm[i], self.Norm[j], self.Norm[i]*self.Norm[j]), file = self.params['DebugFile_file_object'])
            self.model.PCGAngles[AngleIndex][0] = math.acos(ScalProd/((self.Norm[i]*self.Norm[j])+self.Epsilon))
            print("ANGLE IS ", self.model.PCGAngles[AngleIndex][0], file = self.params['EpochValues_file_object'])
            
            #COMPUTATION OF ANGLES FROM NORMS (IT DOESN'T WORKS FOR THE GREAT DIFFERENCE IN THE 2 VECTORS)
            #print('WE HAVE: TOT {}, CLASSES {} {}, SCAL PROD {}'.format(self.TotNormCopy, self.Norm[i]**2, self.Norm[j]**2 ,self.Norm[i]*self.Norm[j] ) )
            #self.model.GradAnglesNormComp[AngleIndex][TimeComp] = (self.TotNormCopy - self.Norm[i]**2 - self.Norm[j]**2)/(2*self.Norm[i]*self.Norm[j])
            #self.model.GradAnglesNormComp[AngleIndex][TimeComp] = math.acos(self.model.GradAnglesNormComp[AngleIndex][TimeComp])
            
            AngleIndex+=1 
                
                
        #WANDB BLOCK

        #create the list of variable to log as table's column
        self.Performance_columns = []
        self.Performance_columns.append('True_Steps')
        self.Performance_columns.append('Resc_Steps')
        self.Grad_columns = []
        self.Grad_columns.append('True_Steps')
        self.Grad_columns.append('Resc_Steps')
        for i in range(0, self.model.num_classes):
            self.Performance_columns.append('Train_Loss_{}'.format(i))
            self.Performance_columns.append('Train_Acc_{}'.format(i))
            self.Grad_columns.append('Grad_batch_Norm_{}'.format(i))
        AngleIndex=0
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):    
            self.Grad_columns.append('Grad_Angle_{}_{}'.format(i,j))

        #log the computed quantities in wandb (also as a table)
        Performance_data = []
        Performance_data.append(1)
        Performance_data.append(self.params['batch_size']/self.params['learning_rate'])
        
        Grad_data = []
        Grad_data.append(1)
        Grad_data.append(self.params['batch_size']/self.params['learning_rate'])
        
        for i in range(0, self.model.num_classes):
            wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][0],
                       'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][0],
                       'Performance_measures/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1})                   
        
            wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(i): (self.model.StepGradientClassNorm[i][0] ),
                       'GradientAngles/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1}) 

            Performance_data.append(self.model.TrainClassesLoss[i][0])
            Performance_data.append(self.model.TrainClassesAcc[i][0])
            Grad_data.append(self.model.StepGradientClassNorm[i][0])

        AngleIndex=0
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
            wandb.log({'GradientAngles/Gradient_Angles_Between_Classes_{}_and_{}'.format(i, j): (self.model.PCGAngles[AngleIndex][0] ),
                       #'GradientAngles/Gradient_Angles_NormComp_Between_Classes_{}_and_{}'.format(i, j): (self.model.GradAnglesNormComp[AngleIndex][Comp] ),
                       'GradientAngles/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1})
            
            Grad_data.append(self.model.PCGAngles[AngleIndex][0])
            AngleIndex+=1 
        
        
        #creation of a table
        self.Performance_data_table = wandb.Table(columns=self.Performance_columns, data=[copy.deepcopy(Performance_data)])
        self.Grad_data_table = wandb.Table(columns=self.Grad_columns, data=[copy.deepcopy(Grad_data)])        
    
    
    
        for EvalKey in self.ValidDL:
            SetFlag = 'Valid' 
            for dataval,labelval in self.ValidDL[EvalKey]:
        
                Mask_Flag = 1
                
                dataval = dataval.double() 
                dataval = dataval.to(self.params['device'])
                labelval = labelval.to(self.params['device']) 

                if self.params['NetMode']=='VGG_Custom_Dropout':
                    
                    self.DropoutBatchForward(dataval, Mask_Flag)
                    Mask_Flag = 0
                else:
                    self.BatchForward(dataval)
                    
                    
                self.output = self.OutDict['out'].clone()

                    
                self.BatchEvalLossComputation(labelval, 0, SetFlag) #computation of the loss function and the gradient (with backward call)

                #computation of quantity useful for precision accuracy,... measures
                self.CorrectBatchGuesses(labelval, 0, SetFlag)
                #Store the last layer mean representation and per classes loss function
                """
                #ADAPT COMP. OF THE LAST LAYER TO THE CASE OF BATCHES (also remember the layer compression line).
                if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                else:
                    NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                """

                self.loss.backward()   # backward pass: compute gradient of the loss with respect to model parameters
                self.GradCopyUpdate(labelval[0]) #again we select one of the label scalar; since all the element in the batch represent the same class they are all equivalent
                self.optimizer.zero_grad()
                    
            #putting gradient to 0 before filling it with the per class normalized sum
            self.optimizer.zero_grad()
            """
            NetInstance.LastLayerRepresCompression()
            """
            
            
        self.LossAccAppend(0, SetFlag)
        
        
        if self.params['ValidMode']=='Test':
    
            for EvalKey in self.TestDL:
                SetFlag = 'Test' 
                for dataval,labelval in self.TestDL[EvalKey]:
            
                    Mask_Flag = 1
                    
                    dataval = dataval.double() 
                    dataval = dataval.to(self.params['device'])
                    labelval = labelval.to(self.params['device']) 
    
                    if self.params['NetMode']=='VGG_Custom_Dropout':
                        
                        self.DropoutBatchForward(dataval, Mask_Flag)
                        Mask_Flag = 0
                    else:
                        self.BatchForward(dataval)
                        
                        
                    self.output = self.OutDict['out'].clone()
    
                        
                    self.BatchEvalLossComputation(labelval, 0, SetFlag) #computation of the loss function and the gradient (with backward call)
    
                    #computation of quantity useful for precision accuracy,... measures
                    self.CorrectBatchGuesses(labelval, 0, SetFlag)
                    #Store the last layer mean representation and per classes loss function
                    """
                    #ADAPT COMP. OF THE LAST LAYER TO THE CASE OF BATCHES (also remember the layer compression line).
                    if NetInstance.params['NetMode']=='VGG_Custom_Dropout':
                        NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                    else:
                        NetInstance.MeanRepresClass[labelval[i]].append(NetInstance.OutDict['l2'].clone().double())
                    """
    
                    self.loss.backward()   # backward pass: compute gradient of the loss with respect to model parameters
                    self.GradCopyUpdate(labelval[0]) #again we select one of the label scalar; since all the element in the batch represent the same class they are all equivalent
                    self.optimizer.zero_grad()
                        
                #putting gradient to 0 before filling it with the per class normalized sum
                self.optimizer.zero_grad()
                """
                NetInstance.LastLayerRepresCompression()
                """    
            self.LossAccAppend(0, SetFlag)  

     


    def wandb_tables_init(self):
        """
        create the list of variable to log as table's column

        Returns
        -------
        None.

        """
        
        self.Performance_columns = []
        self.Performance_columns.append('True_Steps')
        self.Performance_columns.append('Resc_Steps')
        self.Grad_columns = []
        self.Grad_columns.append('True_Steps')
        self.Grad_columns.append('Resc_Steps')
        for i in range(0, self.model.num_classes):
            self.Performance_columns.append('Train_Loss_{}'.format(i))
            self.Performance_columns.append('Train_Acc_{}'.format(i))
            self.Grad_columns.append('Grad_batch_Norm_{}'.format(i))
        AngleIndex=0
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):    
            self.Grad_columns.append('Grad_Angle_{}_{}'.format(i,j))

        """
        #log the computed quantities in wandb (also as a table)
        Performance_data = []
        Performance_data.append(1)
        Performance_data.append(self.params['batch_size']/self.params['learning_rate'])
        
        Grad_data = []
        Grad_data.append(1)
        Grad_data.append(self.params['batch_size']/self.params['learning_rate'])
        
        for i in range(0, self.model.num_classes):
            wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][0],
                       'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][0],
                       'Performance_measures/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1})                   
        
            wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(i): (self.model.StepGradientClassNorm[i][0] ),
                       'GradientAngles/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1}) 

            Performance_data.append(self.model.TrainClassesLoss[i][0])
            Performance_data.append(self.model.TrainClassesAcc[i][0])
            Grad_data.append(self.model.StepGradientClassNorm[i][0])

        AngleIndex=0
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
            wandb.log({'GradientAngles/Gradient_Angles_Between_Classes_{}_and_{}'.format(i, j): (self.model.PCGAngles[AngleIndex][0] ),
                       #'GradientAngles/Gradient_Angles_NormComp_Between_Classes_{}_and_{}'.format(i, j): (self.model.GradAnglesNormComp[AngleIndex][Comp] ),
                       'GradientAngles/Rescaled_Steps': (self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': 1})
            
            Grad_data.append(self.model.PCGAngles[AngleIndex][0])
            AngleIndex+=1 
        
        """
        #creation of a table
        self.Performance_data_table = wandb.Table(columns=self.Performance_columns)
        self.Grad_data_table = wandb.Table(columns=self.Grad_columns)      




        
    #reset some variables used for temp storing of gradient information    
    def StoringGradVariablesReset(self):
        """
        Clear the Norm of the gradient and the temp variable where we store it before assign it to the p.grad

        Returns
        -------
        None.

        """
        self.Norm = np.zeros(self.model.num_classes)
        self.GradCopy = [[] for i in range(self.model.num_classes)] 
            
    def CorrelationTempVariablesInit(self):
        """
        reset Temp variables for correlation computation

        Returns
        -------
        None.

        """
        self.TwWeightTemp = []
        self.OverlapTemp = []
        self.DistanceTemp = []
    
    #reset of the class repr. vector, the total norm and the vectors of correct guess
    def EvaluationVariablesReset(self):
        """
        reset the variable associated to evaluation measures (train and test correct guesses (total and per class))

        Returns
        -------
        None.

        """
             
            
        #reset Gradient Norm (calculated for each epoch)
        self.total_norm = 0        
        #prepare tensor to store the mean representation of the classes
        self.MeanRepresClass = [[] for i in range(self.model.num_classes)]       
        # variables for training accuracy
        self.TrainCorrect = 0
        self.ClassTrainCorrect = np.zeros(self.model.num_classes)        
        #self.TrainTotal = len(self.train_loader.sampler)   
        #since we are dividing the dataloader associated to different classes we calculate the Traintotal from the sum of the classes' dataloader       
        self.TestCorrect = 0
        self.ClassTestCorrect = np.zeros(self.model.num_classes)       
        #self.ValTotal = len(self.valid_loader.sampler)
        self.ValCorrect = 0
        self.ClassValCorrect = np.zeros(self.model.num_classes)  
         
        self.train_loss = 0
        self.valid_loss = 0
        self.test_loss = 0
        
    #this is a single line command; I put it here simply to recall easly the returned variable   
    def BatchForward(self, data):
        """
        forward propagation of the batch (data) across the Net

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.OutDict = self.model(data)
        self.output = self.OutDict['out'].clone()
 
    def DropoutBatchForward(self, data, MF):
        """
        define a method to use in case of models with customized dropout.
        The difference with normal forward is that in this case we have 2 more flag arguments to understand when is time to 
        update the masks and wheter we are in train or eval mode.
        
        Note: every time you write model() you make a forward step, meaning also gradient accumulation. For each step this command should be given only one time, for this reason we save all the output dict in a variable and recall from it instead of reuse model()

        Parameters
        ----------
        SampleData : TYPE
            DESCRIPTION.
        MF : TYPE
            DESCRIPTION.
        TMF : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        self.OutDict = self.model(data, MF)
        self.output = self.OutDict['out'].clone()           
 
    


    def LastLayerRepr(self, data, label):
        """
        storing the mean representation (defined by the last layer weight's vector) for each classes; by this representation you can compute an angle between classes

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        for im in range(0, len(label)):
            #print("THE L2", self.model(data)['l2'][im].shape, label[im])
            self.MeanRepresClass[label[im]].append(self.model(data)['l2'][im].clone().double())
        for index in range(0, self.model.num_classes):
            if self.MeanRepresClass[index]:
                #print("indice ",index, self.MeanRepresClass[index][0].shape, self.MeanRepresClass[index][1].shape )
                self.MRC[index]= sum(torch.stack(self.MeanRepresClass[index])).detach().clone()
                self.MeanRepresClass[index].clear()
                self.MeanRepresClass[index].append(self.MRC[index].detach().clone())            
        
    #TODO: LossComputation and  SampleLossComputation can be joined in a single method       
    def LossComputation(self, label):
        """
        compute the loss

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if(self.params['NetMode']=='Single_HL'):
            #print('test of loss', type(self.output),type(torch.reshape(label, (-1,))))
            #print('test of loss', self.output,torch.reshape(label, (-1,)))
            
            #self.loss = self.MSELoss(self.output.to(torch.float32),torch.reshape(label, (-1,)).to(torch.float32))
            self.loss= self.Simil_HingeLoss(self.output, torch.tensor([self.params['label_to_sign_dict'][x.item()] for x in torch.reshape(label, (-1,))]).to(self.params['device']))
        else:
            # calculate the loss; we use torch.reshape(label, (-1,)) to convert a tensor of singles scalaras into a 1-D tensor (shape required to use criterion method)
            self.loss = self.criterion(self.output,torch.reshape(label, (-1,)))
                            
        #SPHERICAL CONSTRAIN BLOCK
        """
        if (self.params['SphericalConstrainMode']=='ON'):                    
            #add the sphericl regularization constrain to the loss
            #following line constrain the L2 norm to be equal to the number of training parameters (tunable weights)
            #loss += SphericalRegulizParameter*(((sum(q.pow(2.0).sum() for q in model.parameters() if q.requires_grad))-(sum(p.numel() for p in model.parameters() if p.requires_grad))).pow(2.0))
            
            #following line constrain the L2 norm to be equal to the number of neurons (neurons for each layer + number of classes)
            if  (self.params['Dataset']=='MNIST'):
                TotalNeurons = 32 + 32 + torch.count_nonzero(self.train_data.targets.bincount()) 
            elif (self.params['Dataset']=='CIFAR10'):
                TotalNeurons = 32 + 32 + self.model.num_classes
            self.loss += self.params['SphericalRegulizParameter']*(((sum(q.pow(2.0).sum() for q in self.model.parameters() if q.requires_grad))-(TotalNeurons)).pow(2.0))
        """

    def BatchEvalLossComputation(self, label, TimeComp=None, SetFlag=None):
        """
        compute per class loss function and the total one
        if in eval mode the loss is added to the measure of the corresponding dataset
        Parameters
        ----------
        label : vector
            labels of the batch's elements.
        TimeComp : int
            time component of the array (second component).
        SetFlag : string
            specify (for the evaluation mode) which dataset we are using at the moment of the function's call (and so where to store the measures) 

        Returns
        -------
        None.

        """

        if(self.params['NetMode']=='Single_HL'):
            #print('test of loss', type(self.output),type(torch.reshape(label, (-1,))))
            #print('test of loss', self.output,torch.reshape(label, (-1,)))
            
            #self.loss = self.MSELoss(self.output.to(torch.float32),torch.reshape(label, (-1,)).to(torch.float32)) #.float() added becauseself.MSELoss expect same type: if you use the wrong type the error will arise during the backprop command
            self.loss= self.Simil_HingeLoss(self.output, torch.tensor([self.params['label_to_sign_dict'][x.item()] for x in torch.reshape(label, (-1,))]).to(self.params['device']))   
            #print('a loss è', self.loss)
        else:
            self.loss = self.criterion(self.output,torch.reshape(label, (-1,))) #reduction = 'none' is the option to get back a single loss for each sample in the batch (for this reason I use the SampleCriterion)
        #print("label prima", label)
        label = torch.reshape(label, (-1,))
        #print("label dopo", label)
        #print("label[0]", label[0])
        
        #WARNING: we now don't have the problem of mixed classes inside the batch but you have to be carefull about the choice of the criterion mode; we don't want the mean over the batch but the sum (because the batches of dataloader associated to different classes may have a different size (if there is class imbalance))
        #we select one random label[i] (they are all the same) and assign the computed loss to the variable of the corresponding class
        #note that we check to be in eval mode (since both train and eval are computed there) 
        if not self.model.training:
            if SetFlag =='Train':
                self.model.TrainClassesLoss[label[0]][TimeComp] += (self.loss).sum().item()
            elif SetFlag=='Valid':                
                self.model.ValidClassesLoss[label[0]][TimeComp] += (self.loss).sum().item()
            elif SetFlag=='Test':                
                self.model.TestClassesLoss[label[0]][TimeComp] += (self.loss).sum().item()
            else:
                print("WARNING: you set a wrong value for the set flag: only Train Valid and Test are allowed", file = self.params['WarningFile'])
            

        
        """
        #OLD APPROACH WITH ALL THE CLASSES MIXED INTO ONE SINGLE DATALOADER
        for i in range (0, self.model.num_classes):
            #print(((torch.reshape(label, (-1,))==i).int()*self.loss).sum().item(), (torch.reshape(label, (-1,))==i).int(), self.loss )
            vec[i][TimeComp] += ((label==i).int()*self.loss).sum().item() #sum up all term in loss belonging to the same class and add to the corresponding component 
        #print('LABEL LOSS', (label==i).int(), self.loss, ((label==i).int()*self.loss).sum().item()  )
        """
    def SampleLossComputation(self, label):
        """
        compute the loss on a single image of the batch as input

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #reshape output in 1-d vector to make it compatible with criterion input shape
        self.loss = self.criterion(self.output.expand(1, self.model.num_classes),(label).expand(1))
        
        #SPHERICAL CONSTRAIN BLOCK
        """
        if (self.params['SphericalConstrainMode']=='ON'):
            
            #add the sphericl regularization constrain to the loss
            #following line constrain the L2 norm to be equal to the number of training parameters (tunable weights)
            #loss += SphericalRegulizParameter*(((sum(q.pow(2.0).sum() for q in model.parameters() if q.requires_grad))-(sum(p.numel() for p in model.parameters() if p.requires_grad))).pow(2.0))
            
            #following line constrain the L2 norm to be equal to the number of neurons (neurons for each layer + number of classes)
            if  (self.params['Dataset']=='MNIST'):
                TotalNeurons = 32 + 32 + torch.count_nonzero(self.train_data.targets.bincount()) 
            elif (self.params['Dataset']=='CIFAR10'):
                TotalNeurons = 32 + 32 + self.model.num_classes
            self.loss += self.params['SphericalRegulizParameter']*(((sum(q.pow(2.0).sum() for q in self.model.parameters() if q.requires_grad))-(TotalNeurons)).pow(2.0))
        """
            



        
    
    def CorrectBatchGuesses(self, label, TimesComponentCounter, SetFlag):
        """
        Evaluate how many guesses (Between the BS images forwarded ) were correctly assigned (are equal to the real label)

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.
        TimesComponentCounter : TYPE
            DESCRIPTION.
        SetFlag : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if SetFlag=='Train':
            
            if self.params['OutShape']=='Single_Node':
                self.pred=self.OutDict['pred'].clone()
                self.TrainPred = self.pred
            else:
                _, self.TrainPred = torch.max(self.output, 1)
            label = torch.reshape(label, (-1,))
            for i in range(0, self.model.num_classes): #here we have to mantain the cycle (also if all the element in the batch belong to the same class) for the computation of FP
                #print(type(label), type(self.TrainPred))
                #print(label, self.TrainPred)
                self.ClassTrainCorrect[i] += ((label==i).int()*(self.TrainPred==i).int()).sum().item()
                self.TrainCorrect+= ((label==i).int()*(self.TrainPred==i).int()).sum().item()
                self.model.TP[i][TimesComponentCounter] += ((label==i).int()*(self.TrainPred==i).int()).sum().item()
                self.model.FP[i][TimesComponentCounter] += ((label!=i).int()*(self.TrainPred==i).int()).sum().item()
                self.model.FN[i][TimesComponentCounter] += ((label==i).int()*(self.TrainPred!=i).int()).sum().item()  
        elif SetFlag=='Valid':

            _, self.ValPred = torch.max(self.output, 1)
            label = torch.reshape(label, (-1,))
            #we use, as usual, the knowledge that all batch's elements belong to the same class
            i = label[0] 
            self.ClassValCorrect[i] += ((label==i).int()*(self.ValPred==i).int()).sum().item()
            self.ValCorrect+= ((label==i).int()*(self.ValPred==i).int()).sum().item()
        
        elif SetFlag=='Test':
            #calculating correct samples for accuracy
            _, self.TestPred = torch.max(self.output, 1)    
 
            label = torch.reshape(label, (-1,))
            i = label[0]
            self.ClassTestCorrect[i] += ((label==i).int()*(self.TestPred==i).int()).sum().item()
            self.TestCorrect+= ((label==i).int()*(self.TestPred==i).int()).sum().item()       
         

        
    def LossAccAppend(self, TimesComponentCounter, SetFlag):
        """
        Accuracy and Losses (total and per class) are computed and assigned to the corresponding measure variable
        NOTE: 
            -the loss function (both the total and the per-class) is normalized for the number of element over which is calculated
            - for such normalization is sufficient to use the number of elements (in the whole dataset or only in the dataloader corresponding to a  specific class)
             In the oversampled algorithms there is no difference because in any case the evaluation phase check the whole dataset one time (without repetition); in fact we are just evaluating, not forwarding to update

        Parameters
        ----------
        TimesComponentCounter : TYPE
            DESCRIPTION.
        SetFlag : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if SetFlag=='Train':
            self.model.TrainLoss.append(np.sum(self.model.TrainClassesLoss, axis=0)[TimesComponentCounter] / self.TrainTotal)   
            self.model.TrainAcc.append(100*self.TrainCorrect / self.TrainTotal)
                   
            for k in range(self.model.num_classes):
                self.model.TrainClassesAcc[k][TimesComponentCounter] = (100*self.ClassTrainCorrect[k] / self.SamplesClass[k])
                self.model.TrainClassesLoss[k][TimesComponentCounter] = self.model.TrainClassesLoss[k][TimesComponentCounter]/self.SamplesClass[k]
                print("Train Class Loss saved for class {} is {}".format(k, self.model.TrainClassesLoss[k][TimesComponentCounter]), file = self.params['EpochValues_file_object'])
        if SetFlag=='Test':
            self.model.TestLoss.append( np.sum(self.model.TestClassesLoss, axis=0)[TimesComponentCounter]/ self.TestTotal) 
            
            self.model.TestAcc.append(100*self.TestCorrect / self.TestTotal)
            
        
            for k in range(self.model.num_classes):
                self.model.TestClassesAcc[k][TimesComponentCounter] = (100*self.ClassTestCorrect[k] / self.TestSamplesClass[k])
                self.model.TestClassesLoss[k][TimesComponentCounter] = self.model.TestClassesLoss[k][TimesComponentCounter]/self.TestSamplesClass[k]
                print("Test Class Loss saved for class {} is {}".format(k, self.model.TestClassesLoss[k][TimesComponentCounter]), file = self.params['EpochValues_file_object'])
        if SetFlag=='Valid':
            self.model.ValidLoss.append( np.sum(self.model.ValidClassesLoss, axis=0)[TimesComponentCounter]/ self.ValTotal) 
            
            self.model.ValidAcc.append(100*self.ValCorrect / self.ValTotal)
            
        
            for k in range(self.model.num_classes):
                self.model.ValidClassesAcc[k][TimesComponentCounter] = (100*self.ClassValCorrect[k] / self.ValidSamplesClass[k])
                self.model.ValidClassesLoss[k][TimesComponentCounter] = self.model.ValidClassesLoss[k][TimesComponentCounter]/self.ValidSamplesClass[k]
                print("Vaid Class Loss saved for class {} is {}".format(k, self.model.ValidClassesLoss[k][TimesComponentCounter]), file = self.params['EpochValues_file_object'])
                           

        

    def GradCopyUpdate(self, label):
        """
        copy the gradient to a storing variable that will be used to implement the update prescribed by the algorithm

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        ParCount = 0
        if not self.GradCopy[label]:
            for p in self.model.parameters():   
                #self.GradCopy[label].append(p.grad.clone().double()) #we copy the gradient with double precision (.double()) to prevent rounding error in case of very small numbers                           
                self.GradCopy[label].append(p.grad.clone())
                
        elif self.GradCopy[label]:
            for p in self.model.parameters(): 
                self.GradCopy[label][ParCount] = self.GradCopy[label][ParCount].clone() + p.grad.clone()#self.GradCopy[label][ParCount].clone() + p.grad.clone().double()
                ParCount +=1


    def NormalizeGradVec(self):
        """
        normalize the gradient vector

        Returns
        -------
        None.

        """
        Norm =0
        for p in self.model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            Norm += param_norm.item() ** 2   
        for p in self.model.parameters():
            p.grad = torch.div(p.grad.clone(), Norm**0.5)


    def AssignNormalizedTotalGradient(self, TimeComp):
        """
        Manually assign the gradient associated to self.model.parameters() with the normalized sum of classes gradients 

        Parameters
        ----------
        TimeComp : int
            index to save the class norm into an array

        Returns
        -------
        None.

        """
        self.TotGrad = []
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1

            self.Norm[index] = self.Norm[index]**0.5/self.RoundSolveConst
        self.model.ClassesGradientNorm[TimeComp] = self.Norm
            
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2   
            #self.total_norm += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2     
            
        self.TotNormCopy = self.total_norm
        ParCount = 0    
        for p in self.model.parameters():
            #CHOOSE ONE OF THE FOLLOWING NORMALIZATION PROCEDURE FOR THE GRADIENT
            if self.params['StochasticMode']=='OFF': #if we deal with a fully batch algorithm we normalize with the whole dataset size
                p.grad += torch.div(self.TotGrad[ParCount].clone(), self.TrainTotal) #normalize for the number of samples in the training set
                #p.grad += torch.div(self.TotGrad[ParCount].clone(), (((self.total_norm**0.5)/self.RoundSolveConst) + 0.000001)) #normalize using the gradient l-2 norm (we add an infinitesimal regularizer to avoid the 0-division case)
                ParCount +=1  
            elif self.params['StochasticMode']=='ON':#if we use a mini-batch algorithm we normalize by the batch size
                p.grad += torch.div(self.TotGrad[ParCount].clone(), np.sum(self.TrainTotalClassBS)) #TrainClassBS #normalize for the number of samples in the training set
                #p.grad += torch.div(self.TotGrad[ParCount].clone(), (((self.total_norm**0.5)/self.RoundSolveConst) + 0.000001)) #normalize using the gradient l-2 norm (we add an infinitesimal regularizer to avoid the 0-division case)
                ParCount +=1  


    #reset some variables used for temp storing of gradient information    
    def StoringDATASET_GradReset(self):
        """
        Clear the Norm of the gradient and the temp variable where we store it before assign it to the p.grad

        Returns
        -------
        None.

        """
        self.DatasetNorm = np.zeros(self.model.num_classes)
        self.DataGradCopy = [[] for i in range(self.model.num_classes)] 

    def SaveNormalizedGradient(self):
        """
        save the normalized gradient on a second variable to perform the signal projection

        Returns
        -------
        None.

        """
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.DatasetNorm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
    
            self.DatasetNorm[index] = (self.DatasetNorm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                self.DataGradCopy[index].append(torch.div(self.GradCopy[index][ParCount].clone(), (self.DatasetNorm[index] + 0.000001)).clone())
                ParCount +=1   
      

    def SignalProjectionNorm(self):
        """
        projection normalization:
            -we normalize the batch class vector
            -compute their projection along the signal (normalized vector)
            -we moltiply each class vector for the scalr product of the other class such that the 2 will assume the same value
        Returns
        -------
        None.
        """
        self.Norm = np.zeros(self.model.num_classes) #reset norm
        self.SignalProj = np.zeros(self.model.num_classes)
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
            
            #compute batch classes norms to pass from vector to versor
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2        
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            
            #compute the scalar product between the dataset vectors and the batchs' ones
            TGComp=0
            
            for obj in self.GradCopy[index]:

                self.SignalProj[index] += np.sum(np.multiply(self.DataGradCopy[index][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst))


                TGComp +=1
            self.SignalProj[index] = self.SignalProj[index]/(self.RoundSolveConst*self.RoundSolveConst*self.Norm[index])       
        
        #we define the factor to put the same projection value to each class
        self.ProjNormFactor = self.SignalProj.prod()/self.SignalProj
        for index in range(0, self.model.num_classes):
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), (self.Norm[index] + 0.000001))*self.ProjNormFactor[index]
                ParCount +=1               
            
            
            
            
            
            
            

    def PerClassNormalizedGradient(self, TimeComp):
        """
        vector for the weights' update is given by the sum of class gradient terms, each one normalized 

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.TotGrad = []
        
        

        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), (self.Norm[index] + 0.000001))
                ParCount +=1   
        if self.model.training:  # assign the mean gradient norm only in the training phase and not during evaluation cycle   
            self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2        
        self.TotNormCopy = self.total_norm



    def BisectionGradient(self, TimeComp):
        """
        vector for the weights' update is given by the sum of class gradient terms, each one normalized 

        Parameters
        ----------


        Returns
        -------
        None.

        """
        self.TotGrad = []
        self.PCNorm =0
        

        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), (self.Norm[index] + 0.000001))
                ParCount +=1   
        if self.model.training:  # assign the mean gradient norm only in the training phase and not during evaluation cycle   
            self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2        
        self.TotNormCopy = self.total_norm
        
        #calculating the norm of the per class normalized and renormalize with the total norm 
        for p in self.model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            self.PCNorm += param_norm.item() ** 2
        self.PCNorm =  self.PCNorm** 0.5   
        
        for p in self.model.parameters():
            p.grad = torch.mul(torch.div(p.grad.detach(), (self.PCNorm+ 0.000001)), (self.total_norm**0.5))


    def GradNorm(self, TimeComp):
        """
        Compute the norm of gradient associated to each class

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.Norm = np.zeros(self.model.num_classes) #reset norm
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0  
        self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm

    def NormGrad1Copy(self):
        """
        copy the normalized grad at each 'relevant' step.
        call this function after the PerClassNormalizedGradient(self, TimeComp) in which classes norms of gradients are computed

        Returns
        -------
        None.

        """
        
        self.NormGrad1 = [[] for i in range(self.model.num_classes)] 
        self.Grad1_Norm = np.zeros(self.model.num_classes)
        
        if(all(self.GradCopy)):  
            #self.ns+=1
            for index in range(0, self.model.num_classes):# we normalize fixing the norm to self.RoundSolveConst to avoid to much lit values (underflow issues during overlap)
                for obj in self.GradCopy[index]:
                    self.Grad1_Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2    
                self.Grad1_Norm[index] = (self.Grad1_Norm[index]**0.5)/self.RoundSolveConst    
                ParCount=0
                for p in self.model.parameters():
                    self.NormGrad1[index].append(torch.div((copy.deepcopy(self.GradCopy[index][ParCount])*self.RoundSolveConst), (self.Grad1_Norm[index]+ 0.00000001)))
                    ParCount+=1
            self.NormGrad1Tot.append(copy.deepcopy(self.NormGrad1))
                
    def NormGrad2Copy(self):
        """
        Copy the normalized gradient after the 'relevant' step. Here I have to compute the norm also because is not stored in 
        during a call of a different method (self.Norm is used for NormGrad1Copy )
        ; and compute the overlap with the previous one
        

        Returns
        -------
        None.

        """
        self.NormGrad2 = [[] for i in range(self.model.num_classes)] 
        #self.NormGradOverlap = np.zeros((self.model.num_classes, self.params['samples_grad']))
        self.NormGradOverlap = np.zeros((self.model.num_classes, len(self.train_loader)))
        self.Grad2_Norm = np.zeros(self.model.num_classes)

        if(all(self.GradCopy) and all(self.NormGrad1)):
            #self.ns+=1
            for index in range(0, self.model.num_classes):
                #compute the total loss function as the sum of all class losses
                for obj in self.GradCopy[index]:
                    self.Grad2_Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2    
                self.Grad2_Norm[index] = (self.Grad2_Norm[index]**0.5)/self.RoundSolveConst
                for obj in self.GradCopy[index]:
                    self.NormGrad2[index].append(torch.div((obj.cpu().clone()*self.RoundSolveConst), (self.Grad2_Norm[index]+ 0.00000001)))
                
                
            self.NormGrad2Tot.append(copy.deepcopy(self.NormGrad2))
                


    def Wandb_Log_Grad_Overlap(self, TimeStep):
        """
        logging on wandb the measure of gradient overlap between classes

        Parameters
        ----------
        TimeStep : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        print("the length of the lists is {} and {}".format(len(self.NormGrad1Tot), len(self.NormGrad2Tot) ))
        
        if(all(self.GradCopy) and all(self.NormGrad1)):
            for i in range(0, len(self.NormGrad2Tot)):
                for index in range(0, self.model.num_classes):               
                
                    #overlap computation
                    ParCount=0
                    for p in self.model.parameters():
                        #RIPARTI DA QUA
                        self.NormGradOverlap[index][i] += np.sum(np.multiply((self.NormGrad1Tot[0][index][ParCount].cpu().detach().numpy()), (self.NormGrad2Tot[i][index][ParCount].cpu().detach().numpy())))
                        ParCount +=1     
                    self.NormGradOverlap[index][i] = self.NormGradOverlap[index][i]/(self.RoundSolveConst*self.RoundSolveConst)

            
            for i in range(0, self.model.num_classes):
                wandb.log({'Performance_measures/Normalized_Gradient_Overlap_Class_{}'.format(i): np.mean(self.NormGradOverlap,1)[i],
                           'Performance_measures/Rescaled_Steps': (TimeStep*self.params['batch_size']/self.params['learning_rate']),
                           'Performance_measures/True_Steps_+_1': TimeStep+1}) 
                
        self.NormGrad1Tot = []
        self.NormGrad2Tot = []     

        
               
                
    def PerClassMeanGradient(self, TimeComp):
        """
        Instead of normalize the per class gradient with the norm (as in PerClassNormalizedGradient) here we divide for the number of elements

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.
            
        class_members:vector
            store the number of element for each batch (we use them to normalize the per class components).
            To use the algorithm for PCNGD give in input the total number of members in the dataset

        Returns
        -------
        None.

        """
        self.TotGrad = []

        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            TGComp=0
            for obj in self.GradCopy[index]:
                self.Norm[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2
                
                #filling the total gradient
                if(index==0):
                    self.TotGrad.append(obj.clone())
                else:
                    self.TotGrad[TGComp] += obj.clone()
                    TGComp+=1
    
            self.Norm[index] = (self.Norm[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad += torch.div(self.GradCopy[index][ParCount].clone(), self.TrainClassBS[index])
                ParCount +=1   
        self.model.ClassesGradientNorm[TimeComp] += self.Norm #for the PCNGD I have only one element for each epoch, for the PCNSGD I have one for each batch; I sum up all of them (and divide for the number of batches to obtain an average value)                
        #calculating the total gradient norm
        for obj in self.TotGrad:
            self.total_norm += (torch.norm(obj.cpu().clone()).detach().numpy())**2 
        self.TotNormCopy = self.total_norm


    def PCN_lr_scheduler(self):
        """
        modify the learning rate during the simulation according to LR->LR*(1+cos(a)), where alpha is the angle between the 2 classes gradient

        Returns
        -------
        None.

        """
        
        cos_alpha = self.cos_alpha           


        lr = self.params['learning_rate']*(1+cos_alpha)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



    def StepSize(self):
        """
        compute the size of forwarding steps (norm of the gradient*learning rate)

        Returns
        -------
        None.

        """
        self.Step=0
        for p in self.model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            self.Step += param_norm.item() ** 2             
        print(self.Step**0.5, flush=True, file = self.params['StepNorm_file_object'])
        
    def GradientAngles(self, TimeComp):
        """
        Calculate the gradient angles between classes and their norm

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if(all(self.GradCopy)): #I add this condition to be sure there is at least 1 element of each class (for the SGD case)
            AngleIndex=0
            for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                #calculate the cos(ang) as the scalar product normalized with the l2 norm of the vectors
                ScalProd=0
    
                TGComp=0
                for obj in self.GradCopy[i]:
    
                    #print("GRAD COMP i {}".format(obj[0]))
                    #print("GRAD COMP j {}".format(self.GradCopy[j][TGComp][0]))
                    #print("DOT pROD {} E LA SOMMA {}".format(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy(), obj.cpu().clone().detach().numpy())[0], np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy(), obj.cpu().clone().detach().numpy()))))
                    #print("CON I LONG DOUBLE {} E LA SOMMA {}".format(np.multiply(np.longdouble(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()), np.longdouble(obj.cpu().clone().detach().numpy()))[0], np.sum(np.multiply( np.longdouble(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()), np.longdouble(obj.cpu().clone().detach().numpy())))) )
                    ScalProd += np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst))

                    """
                    if TGComp==0:
                        print("IL TIPO INCRIMINATO È", obj.type())
                        print("L'ORDINE DI GRANDEZZA", obj, obj.cpu().clone().detach().numpy())
                        print("L'ALTRO", self.GradCopy[j][TGComp], self.GradCopy[j][TGComp].cpu().clone().detach().numpy())
                        print("L'ORDINE DOPO IL DOT PROD", np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst) )
                    print("LA SOMMA", np.sum(np.multiply(self.GradCopy[j][TGComp].cpu().clone().detach().numpy()*self.RoundSolveConst, obj.cpu().clone().detach().numpy()*self.RoundSolveConst)))
                    """
                    TGComp +=1
                ScalProd = ScalProd/(self.RoundSolveConst*self.RoundSolveConst)       
                
                #saving angle between 2 classes in a variable used for the lr_rate schedule
                #TODO: the following works only for the 2 classes problem (only one angle) pay attention in passing to more classes
                self.cos_alpha = ScalProd/((self.Norm[i]*self.Norm[j])+self.Epsilon) #I add a regularizzation in case I get near pi to preventh math error domain due to rounding errors 
                
                print("The Scalar product between class {} and {} and corresponding norms are: {} {} {}, il prodotto delle 2 norme {}".format(i,j,ScalProd, self.Norm[i], self.Norm[j], self.Norm[i]*self.Norm[j]), flush=True)
                print("The Scalar product between class {} and {} and corresponding norms are: {} {} {}, il prodotto delle 2 norme {}".format(i,j,ScalProd, self.Norm[i], self.Norm[j], self.Norm[i]*self.Norm[j]), file = self.params['DebugFile_file_object'])
                self.model.PCGAngles[AngleIndex][TimeComp+1] = math.acos(ScalProd/((self.Norm[i]*self.Norm[j])+self.Epsilon)) #I add a regularizzation in case I get near pi to preventh math error domain due to rounding errors 
                print("ANGLE IS ", self.model.PCGAngles[AngleIndex][0], file = self.params['EpochValues_file_object'])
                
                #COMPUTATION OF ANGLES FROM NORMS (IT DOESN'T WORKS FOR THE GREAT DIFFERENCE IN THE 2 VECTORS)
                #print('WE HAVE: TOT {}, CLASSES {} {}, SCAL PROD {}'.format(self.TotNormCopy, self.Norm[i]**2, self.Norm[j]**2 ,self.Norm[i]*self.Norm[j] ) )
                #self.model.GradAnglesNormComp[AngleIndex][TimeComp] = (self.TotNormCopy - self.Norm[i]**2 - self.Norm[j]**2)/(2*self.Norm[i]*self.Norm[j])
                #self.model.GradAnglesNormComp[AngleIndex][TimeComp] = math.acos(self.model.GradAnglesNormComp[AngleIndex][TimeComp])
                
                AngleIndex+=1 
            for k in range(0, self.model.num_classes):
                self.model.StepGradientClassNorm[k][TimeComp+1] = self.Norm[k]
                

            


    def LastLayerRepresCompression(self):
        """
        We compress (sum up) the components calculated over the elements of a batch during the step.
        The compression happen as follows:
            - tensors of batch's in MeanRepresClass elements are summed up togheter
            - the resulting summed tensor is charged on MRC or added to it (depending  if MRC is empty or not)
            - MeanRepresClass is cleared to host the next batch's tensor
        Returns
        -------
        None.

        """
        for index in range(0, self.model.num_classes):
            if self.MeanRepresClass[index]:
                
                self.MeanRepresClass[index][0] = torch.sum(self.MeanRepresClass[index][0], dim=0) #sum up all the batch's elements; index 0 is due to the only presence of one batch tensor in the list
                #self.MRC[index]= sum(torch.stack(self.MeanRepresClass[index])).detach().clone()
                if not self.MRC[index]:#if the MRC[index] is empty (i.e. is the first batch with that index) we move the MeanRepres calculated over the batch there
                    self.MRC[index].append(self.MeanRepresClass[index][0].detach().clone())
                elif self.MRC[index]: #if, on the other hand, there is already something inside we add the old tensor to the new one calculated over the last batch forwarded
                    self.MRC[index][0] = torch.add(self.MRC[index][0].detach().clone(), self.MeanRepresClass[index][0].detach().clone())
                self.MeanRepresClass[index].clear() #we clear the MeanRepresClass for the next step
                
                #self.MeanRepresClass[index].append(self.MRC[index].detach().clone()) 
        

    def ReprAngles(self, TimeComp):
        """
        Compute the angles between the last layer representation of the classes (see LastLayerRepr)

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #first we calculate the mean representation of the various classes    
        for index in range(0, self.model.num_classes):
            self.MRC[index][0] = torch.div(self.MRC[index][0], self.SamplesClass[index]) 
            #we put the normalization of classes loss in the appensing functions
            #self.model.TrainClassesLoss[index][TimeComp+1] = self.model.TrainClassesLoss[index][TimeComp+1]/self.SamplesClass[index]
            #self.model.TestClassesLoss[index][TimeComp+1] = self.model.TestClassesLoss[index][TimeComp+1]/self.TestSamplesClass[index]
            self.model.RepresentationClassesNorm[index][TimeComp+1] = torch.norm(self.MRC[index][0].cpu()*self.RoundSolveConst).detach().numpy()
            self.model.RepresentationClassesNorm[index][TimeComp+1] = self.model.RepresentationClassesNorm[index][TimeComp+1]/self.RoundSolveConst
        
        #you can write the following iteration in more compact way:
        #from itertools import combinations
        #for i, j in combinations(range(N), 2):
        AngleIndex=0
        if(TimeComp==0):
            print("lista delle classi usate per gli angoli", flush=True, file=self.params['info_file_object'])
        for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
            if(TimeComp==0):
                print(i,j, flush=True, file=self.params['info_file_object'])
            

            #calculate the cos(ang) as the scalar product normalized with the l2 norm of the vectors
            self.model.TrainAngles[AngleIndex][TimeComp+1] = math.acos(np.sum(np.multiply(self.MRC[i][0].cpu().detach().numpy()*self.RoundSolveConst, self.MRC[j][0].cpu().detach().numpy()*self.RoundSolveConst))/((self.model.RepresentationClassesNorm[i][TimeComp+1]*self.model.RepresentationClassesNorm[j][TimeComp+1]*self.RoundSolveConst*self.RoundSolveConst)+self.Epsilon))
            AngleIndex+=1 
            print("angles are:", self.model.TrainAngles[:, TimeComp+1], flush=True, file =self.params['DebugFile_file_object'])

            
    def UpdatePerformanceMeasures(self, TimeComp):
        """
        Compute Precision,Recall and F1-measure

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        #computing precision, recall and F-measure for each class
        self.model.Prec[:, TimeComp] = self.model.TP[:, TimeComp]/ (self.model.TP[:, TimeComp]+self.model.FP[:, TimeComp])
        #self.RealPositive[:, TimeComp] = self.model.TP[:, TimeComp] + self.model.FN[:, TimeComp]
        self.model.Recall[:, TimeComp] = self.model.TP[:, TimeComp]/ (self.model.TP[:, TimeComp] + self.model.FN[:, TimeComp])
        #self.model.PR[:, TimeComp] = self.model.Prec[:, TimeComp]*self.model.Recall[:, TimeComp]
        #self.model.PR[:, TimeComp] = 2*self.model.PR[:, TimeComp]
        self.model.FMeasure[:, TimeComp] = 2*(self.model.Prec[:, TimeComp]*self.model.Recall[:, TimeComp])/ (self.model.Prec[:, TimeComp]+self.model.Recall[:, TimeComp])  

        
    def WeightsForCorrelations(self):
        """
        we save the weight configuration associated with the first time of the 2 involved  in the 2 point correlation/overlap 
        

        Returns
        -------
        None.

        """
        for param in self.model.parameters():
            self.TwWeightTemp.append(param.data)
        self.model.TwWeights.append(copy.deepcopy(self.TwWeightTemp))
        self.TwWeightTemp = []
            
        
    def CorrelationsComputation(self, IterationCounter, N, CorrTimes, tw, t):
        """
        COMPUTATION OF BOTH THE 2 TIMES CORRELATIONS: 
            we use as first time vector the one stored in self.model.TwWeights with WeightsForCorrelations
            and as second a
        

        Parameters
        ----------
        IterationCounter : TYPE
            DESCRIPTION.
        N : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        for a in range(0, self.model.Ntw):
            for b in range(0, self.model.Nt):
                if (IterationCounter==CorrTimes[a][b]):
                    
                    
                    #carico i parametri nuovi sulla variabile d'appoggio
                    for param in self.model.parameters():
                        self.TwWeightTemp.append(param.data)
                    for c in range(0, len(self.model.TwWeights[a])):
                        self.model.TwoTimesOverlap[a][b] += np.sum(np.multiply((self.model.TwWeights[a][c].cpu().detach().numpy()*self.RoundSolveConst), (self.TwWeightTemp[c].cpu().detach().numpy()*self.RoundSolveConst)))
                        self.model.TwoTimesDistance[a][b] += np.sum(np.square((np.subtract((self.TwWeightTemp[c].cpu().detach().numpy()*self.RoundSolveConst), (self.model.TwWeights[a][c].cpu().detach().numpy()*self.RoundSolveConst)))))
                    self.model.TwoTimesOverlap[a][b] = self.model.TwoTimesOverlap[a][b] / N
                    self.model.TwoTimesOverlap[a][b] = self.model.TwoTimesOverlap[a][b] /(self.RoundSolveConst*self.RoundSolveConst)
                    self.model.TwoTimesDistance[a][b] = self.model.TwoTimesDistance[a][b] / N     
                    self.model.TwoTimesDistance[a][b]=self.model.TwoTimesDistance[a][b]/(self.RoundSolveConst*self.RoundSolveConst)
                    #saving correlation on the tensorboard summary
                    #self.writer.add_scalar('Mean square distance for tw = {}'.format(tw[a]), self.model.TwoTimesDistance[a][b], global_step =  t[b])
                    #self.writer.add_scalar('Overlap for tw = {}'.format(tw[a]), self.model.TwoTimesOverlap[a][b], global_step =  t[b])
      
                    self.Corrwriter.add_scalar('Mean square distance for tw = {}'.format(tw[a]), self.model.TwoTimesDistance[a][b], global_step =  t[b])
                    self.Corrwriter.add_scalar('Overlap for tw = {}'.format(tw[a]), self.model.TwoTimesOverlap[a][b], global_step =  t[b])
                    
                    
                    #saving correlation for wandb
                    if (a==(self.model.Ntw-1)):
                        for ftc in range(0, self.model.Ntw):
                            wandb.log({'MeanSquare_Distance/tw_{}'.format(tw[ftc]): self.model.TwoTimesDistance[ftc][b],
                                       'Overlap/tw_{}'.format(tw[ftc]): self.model.TwoTimesOverlap[ftc][b],
                                       'MeanSquare_Distance/t' : t[b],
                                       'Overlap/t' : t[b]})

                    
                    
                    #empty TwWeightTempafter the computation (in order to be used for appending the first time vector in the next WeightsForCorrelations call)
                    self.TwWeightTemp = [] #note the indent; we empty the temp variable here because a single time can be a second corr. time for multiple matrix element (multiple choices of a,b)




    def WeightNormComputation(self):
        """
        compute the norm of the weight of the network

        Returns
        -------
        None.

        """
        self.model.WeightNorm.append( ((sum(p.pow(2.0).sum() for p in self.model.parameters() if p.requires_grad)) / (sum(p.numel() for p in self.model.parameters() if p.requires_grad))) )
 
    

    def UpdateFileData(self):
         """
         Rewrite data saved in files with new rupdated measures 

         Returns
         -------
         None.

         """
        #if the spherical constrain is active we add to the loss a term containing a tensor; so the loss become a tensor itself;
        #since tensor contains additive components with respect to numpy array contain ,components related to gradient computation, to consider only the array values we have to use the detach module
        

         #if the simulation starts from 0 we simply call periodically the update of files  rewriting each time the updated vectors

         #to have the same format for each vector we transpose some of the variables
         """
         if(self.params['SphericalConstrainMode']=='ON'):
             self.TempTrainingLoss=[t.detach().numpy() for t in self.model.TrainLoss]
         elif(self.params['SphericalConstrainMode']=='OFF'):
             self.TempTrainingLoss=self.model.TrainLoss
         """
         self.TempTrainingLoss=self.model.TrainLoss
         with open(self.params['FolderPath'] + "/TrainLoss.txt", "w") as f:
             np.savetxt(f, np.array(self.TempTrainingLoss), delimiter = ',')
         """
         if(self.params['SphericalConstrainMode']=='ON'):    
             self.TempTestLoss=[t.detach().numpy() for t in self.model.TestLoss]  
         elif(self.params['SphericalConstrainMode']=='OFF'):
             self.TempTestLoss=self.model.TestLoss
         """
         
         self.TempValidLoss=self.model.ValidLoss
         with open(self.params['FolderPath'] + "/ValidLoss.txt", "w") as f:
             np.savetxt(f, np.array(self.TempValidLoss), delimiter = ',')
         
         with open(self.params['FolderPath'] + "/TrainAcc.txt", "w") as f:
             np.savetxt(f, self.model.TrainAcc, delimiter = ',') 
         print("the train accuracy saved from the first simulation is ", self.model.TrainAcc)
         with open(self.params['FolderPath'] + "/ValidAcc.txt", "w") as f:
             np.savetxt(f, self.model.ValidAcc, delimiter = ',') 
                     
         with open(self.params['FolderPath'] + "/TrainPrecision.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.Prec), delimiter = ',')
         with open(self.params['FolderPath'] + "/TrainRecall.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.Recall), delimiter = ',')
         with open(self.params['FolderPath'] + "/TrainF_Measure.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.FMeasure), delimiter = ',')
         
         
             
         with open(self.params['FolderPath'] + "/TrainClassesLoss.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TrainClassesLoss), delimiter = ',')    
         with open(self.params['FolderPath'] + "/TrainClassesAcc.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TrainClassesAcc) , delimiter = ',')   
         with open(self.params['FolderPath'] + "/ValidClassesLoss.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.ValidClassesLoss), delimiter = ',')               
         with open(self.params['FolderPath'] + "/ValidClassesAcc.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.ValidClassesAcc) , delimiter = ',')  
             
 
         if self.params['ValidMode']=='Test':        
             self.TempTestLoss=self.model.TestLoss
             with open(self.params['FolderPath'] + "/TestLoss.txt", "w") as f:
                 np.savetxt(f, np.array(self.TempTestLoss), delimiter = ',')
             with open(self.params['FolderPath'] + "/TestAcc.txt", "w") as f:
                 np.savetxt(f, self.model.TestAcc, delimiter = ',')                      
             #per classes accuracy and loss for the test set (for now implemented only for the Gd and PCNGD)    
             with open(self.params['FolderPath'] + "/TestClassesLoss.txt", "w") as f:
                 np.savetxt(f,  np.transpose(self.model.TestClassesLoss), delimiter = ',')       
             with open(self.params['FolderPath'] + "/TestClassesAcc.txt", "w") as f:
                 np.savetxt(f,  np.transpose(self.model.TestClassesAcc) , delimiter = ',')  
          
             
          
         """   
         if(self.params['Dynamic']=='PCNGD'): 
             with open(self.params['FolderPath'] + "/PCGradientAngles.txt", "w") as f:
                 np.savetxt(f, self.model.PCGAngles, delimiter = ',') 
         elif(self.params['Dynamic']=='GD'): 
             with open(self.params['FolderPath'] + "/GDGradientAngles.txt", "w") as f:
                 np.savetxt(f, np.transpose(self.model.PCGAngles), delimiter = ',') 
        """
         with open(self.params['FolderPath'] + "/GradientAngles.txt", "w") as f:
             np.savetxt(f, np.transpose(self.model.PCGAngles), delimiter = ',')       
             
         with open(self.params['FolderPath'] + "/TrainAngles.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TrainAngles), delimiter = ',') 
         with open(self.params['FolderPath'] + "/TestAngles.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.TestAngles), delimiter = ',') 
         
         with open(self.params['FolderPath'] + "/RepresentationNorm.txt", "w") as f:
             np.savetxt(f,  np.transpose(self.model.RepresentationClassesNorm) , delimiter = ',')     
             
         with open(self.params['FolderPath'] + "/GradientNorm.txt", "w") as f:
             np.savetxt(f, np.array(self.model.GradientNorm), delimiter = ',')    

         with open(self.params['FolderPath'] + "/GradientClassesNorm.txt", "w") as f:
             np.savetxt(f, self.model.ClassesGradientNorm , delimiter = ',')   
         
         

         """
         with open(self.params['FolderPath'] + "/TotP.txt", "w") as f:
             np.savetxt(f,  np.transpose((self.model.TP+self.model.FP)), delimiter = ',')   
         self.TempWeightNorm=[t.detach().numpy() for t in self.model.WeightNorm]  
         with open(self.params['FolderPath'] + "/WeightSquaredNorm.txt", "w") as f:
             np.savetxt(f, np.array(self.TempWeightNorm), delimiter = ',')
         """
             
         
         
         
         
         """
         #salva in un file i parametri (Numero di campioni, Numero di tempi, Numero di classi)
         #NOTE: CONTROLLA CHE IL NUMERO DI CLASSI SIA CARICATO CORRETTAMENTE; ClassesNumber
         Parameters = []
         Parameters.append(NSteps)
         Parameters.append(ClassesNumber)
         with open('./'+ args.FolderName + "/Parameters.txt", "w") as f:
             np.savetxt(f, Parameters, delimiter = ',')                 
         """
            
            


                    
                    
        
    def SimulationID(self):
        """
        save some useful information about the simulation in a file (that will be loaded also in wandb)

        Returns
        -------
        None.

        """
        #the server associated to one simulation is a useful information (if for example you run multiple project in parallel and xtake track of the output from an external source (like wandb))
        subprocess.run('echo The server that host the run is:  $(whoami)@$(hostname)', shell=True, stdout=self.params['info_file_object']) 
        subprocess.run('echo The path of the run codes, inside the server, is:  $(pwd)', shell=True, stdout=self.params['info_file_object']) 
        print("The PID of the main code is: ", os.getpid())
        print("The CheckMode is set on {}; when set on 'ON' mode the random seeds of the simulation are set to a fixed value to reproduce the same result over different runs".format(self.params['CheckMode']))
        print('The simulation total time is {} epochs, the start mode is set on {}, we took measures at {} logaritmically equispaced steps/epochs'.format(self.params['n_epochs'], self.params['StartMode'], self.params['NSteps']), file = self.params['info_file_object'])
        print("the simulation use {} as algorithm".format(self.params['Dynamic']), file = self.params['info_file_object'])
        print("the simulation run over the device: ", self.params['device'],  file = self.params['info_file_object'])
        print('The batch size of the simulation is {}, the learning rate is {}, dropout dropping probability is {}, the parameter for group norm is {}'.format(self.params['batch_size'], self.params['learning_rate'], self.params['dropout_p'], self.params['group_factor']), file = self.params['info_file_object'])
        print('The spherical regularization parameter (if the constrain is imposed) is: ', self.params['SphericalRegulizParameter'], file = self.params['info_file_object'])
        print("dataset used in the simulation is: ",self.params['Dataset'], file = self.params['info_file_object'])
        print("classes defined for the simulation are {} ".format(self.params['label_map']), file = self.params['info_file_object'])
        print("the oversampling (i.e. if the batch respect the dataset proportion (OFF) or takes an equal number of element from each class) mode is set on {}".format(self.params['OversamplingMode']), file = self.params['info_file_object'])
        print(" Imbalance factor introduced inside the dataset is {}, in particular the disproportion between the mapped classes is {} ".format(self.params['ClassImbalance'], self.params['ImabalnceProportions']), flush = True, file = self.params['info_file_object'])
        print("architecture used in the simulation is: ",self.params['NetMode'],  file = self.params['info_file_object'])
        print("the size of the batch was ", self.params['batch_size'], ", the learning rate value: ", self.params['learning_rate'], flush = True, file = self.params['info_file_object']) #we flush the printing at the the end
        
        
        
    def TrashedBatchesReset(self):
        self.TrashedBatches=0
        
        
    def CheckClassPresence(self, label):
        """
        check the presence of all the classes inside the batch that is about to be forwarded

        Parameters
        ----------
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.TrainClassBS = np.zeros(self.model.num_classes)
        for im in range(0, len(label)):
            self.TrainClassBS[label[im]] +=1
        self.ClassPresence = np.prod(self.TrainClassBS)        
        
    def SummaryWriterCreation(self, path):
        """
        tensorboard writer creation

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        #self.writer = SummaryWriter(path)
        
        #SUMMARY DIVISION TO DIVIDE INFO TO UPLOAD TO WANDB
        self.writer = SummaryWriter(path+'/NotWandB')
        #I create a separate folder to save correlations (because I want to learn only these in wandb)
        CorrPath = path + '/Corr'
        self.Corrwriter = SummaryWriter(CorrPath)
        
        
    def SummaryScalarsSaving(self, TimeVec, Comp):
        """
        saving summary (measures) on tensorboard

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.params['ValidMode']=='Valid':
            self.writer.add_scalar('Training loss', np.array(self.model.TrainLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid loss', np.array(self.model.ValidLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Training accuracy', (100*self.TrainCorrect / self.TrainTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid accuracy', (100*self.ValCorrect / self.ValTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Balanced Training accuracy', np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes, global_step =  TimeVec[Comp])
            
            for i in range(0, self.model.num_classes):
                self.writer.add_scalar('Training loss class {}'.format(i), self.model.TrainClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Valid loss class {}'.format(i), self.model.ValidClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])                        
                self.writer.add_scalar('Training accuracy class {}'.format(i), self.model.TrainClassesAcc[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Valid accuracy class {}'.format(i), self.model.ValidClassesAcc[i][Comp], global_step =  TimeVec[Comp]) 
     
            #NUOVI CHECK
            """
            if(np.shape(self.model.TwoTimesOverlap)[0]<Comp):
                self.writer.add_scalar('2-Time Autocorrelation', (self.model.TwoTimesDistance[Comp][0]), global_step =  TimeVec[Comp])
                self.writer.add_scalar('Overlap', (self.model.TwoTimesOverlap[Comp][0]), global_step =  TimeVec[Comp])
            """
            
            if self.params['Dynamic']=='SGD':
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), ((ciccia.pow(2.0).sum())**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
            else:
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    self.writer.add_scalar('Gradient Angles between classes {} and {}'.format(i, j), (self.model.PCGAngles[AngleIndex][Comp+1] ), global_step =  TimeVec[Comp])
                    
                    #print("ANGLE IS ", PCGAngles)
                    AngleIndex+=1 
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), (ciccia.pow(2.0).sum())**0.5, global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5), global_step =  TimeVec[Comp])           
                
        elif self.params['ValidMode']=='Test':
            self.writer.add_scalar('Training loss', np.array(self.model.TrainLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid loss', np.array(self.model.ValidLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Test loss', np.array(self.model.TestLoss)[-1], global_step =  TimeVec[Comp])
            self.writer.add_scalar('Training accuracy', (100*self.TrainCorrect / self.TrainTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Test accuracy', (100*self.TestCorrect / self.TestTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Valid accuracy', (100*self.ValCorrect / self.ValTotal), global_step =  TimeVec[Comp])
            self.writer.add_scalar('Balanced Training accuracy', np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes, global_step =  TimeVec[Comp])
            
            for i in range(0, self.model.num_classes):
                self.writer.add_scalar('Training loss class {}'.format(i), self.model.TrainClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Test loss class {}'.format(i), self.model.TestClassesLoss[i][Comp+1], global_step =  TimeVec[Comp])                        
                self.writer.add_scalar('Valid loss class {}'.format(i), self.model.ValidClassesLoss[i][Comp+1], global_step =  TimeVec[Comp]) 
                self.writer.add_scalar('Training accuracy class {}'.format(i), self.model.TrainClassesAcc[i][Comp+1], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Test accuracy class {}'.format(i), self.model.TestClassesAcc[i][Comp], global_step =  TimeVec[Comp]) 
                self.writer.add_scalar('Valid accuracy class {}'.format(i), self.model.ValidClassesAcc[i][Comp], global_step =  TimeVec[Comp]) 
     
            #NUOVI CHECK
            """
            if(np.shape(self.model.TwoTimesOverlap)[0]<Comp):
                self.writer.add_scalar('2-Time Autocorrelation', (self.model.TwoTimesDistance[Comp][0]), global_step =  TimeVec[Comp])
                self.writer.add_scalar('Overlap', (self.model.TwoTimesOverlap[Comp][0]), global_step =  TimeVec[Comp])
            """
            
            if self.params['Dynamic']=='SGD':
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), ((ciccia.pow(2.0).sum())**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5)*self.params['batch_size'], global_step =  TimeVec[Comp])
            else:
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    self.writer.add_scalar('Gradient Angles between classes {} and {}'.format(i, j), (self.model.PCGAngles[AngleIndex][Comp+1] ), global_step =  TimeVec[Comp])
                    
                    #print("ANGLE IS ", PCGAngles)
                    AngleIndex+=1 
                for name, ciccia in self.model.named_parameters():
                    self.writer.add_scalar('Gradient Norm of the layer {}'.format(name), (ciccia.pow(2.0).sum())**0.5, global_step =  TimeVec[Comp])
                self.writer.add_scalar('Step Norm', (self.Step**0.5), global_step =  TimeVec[Comp])     
            
    def PerClassNormGradDistrSaving(self, TimeVec, Comp):
        """
        save on tensorboard the component distribution of gradient associated to each class

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        Norma = np.zeros(self.model.num_classes)
        for index in range(0, self.model.num_classes):
            #compute the total loss function as the sum of all class losses
                   
            for obj in self.GradCopy[index]:
                Norma[index] += (torch.norm(obj.cpu().clone()*self.RoundSolveConst).detach().numpy())**2

    
            Norma[index] = (Norma[index]**0.5)/self.RoundSolveConst
            ParCount = 0
            for p in self.model.parameters():
                p.grad = torch.div(self.GradCopy[index][ParCount].clone(), (Norma[index] + 0.000001))
                ParCount +=1   
     
            for name, ciccia in self.model.named_parameters():

                self.writer.add_histogram('Gradient layer {} of class {}'.format(name, index), ciccia.grad, global_step=TimeVec[Comp])       
            self.optimizer.zero_grad() #clear gradient before passing to the next class 
        self.optimizer.zero_grad() #clear gradient before passing to the next class 
           
    def SummaryDistrSavings(self, TimeVec, Comp):  
        """
        here we save the distribution of weight and gradient on tensorboard (this is an automized useful feature of tensorboard, not present, as far as I know, in wandb)

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        for name, ciccia in self.model.named_parameters():

            #print(ciccia)
            self.writer.add_histogram('Weights layer {}'.format(name), ciccia, global_step=TimeVec[Comp])
            self.writer.add_histogram('Gradient layer {}'.format(name), ciccia.grad, global_step=TimeVec[Comp])
       
    def SummaryHP_Validation(self, lr, bs):
        """
        here we select the measure we want to store to evaluate the right HP (HP tuning) throught tensorboard

        Parameters
        ----------
        lr : TYPE
            DESCRIPTION.
        bs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.writer.add_hparams({'lr': lr, 'bsize': bs}, {'test loss': np.sum(self.model.TestClassesLoss, axis=0)[-1], 'test accuracy': self.model.TestAcc[-1]})
            

    def WandB_logs(self, TimeVec, Comp):
        """
        here we log the relevant measures in wandb

        Parameters
        ----------
        TimeVec : TYPE
            DESCRIPTION.
        Comp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        #TODO: for some reson the plot of classes valid accuracyis shifted forward in the wandb charts; this doesn't seems to happen for the training; fix this logging issue
        if self.params['ValidMode']=='Valid':
            wandb.log({'Performance_measures/Training_Loss': np.array(self.model.TrainLoss)[-1],
                       'Performance_measures/Valid_Loss': np.array(self.model.ValidLoss)[-1],
                       'Performance_measures/Training_Accuracy': (100*self.TrainCorrect / self.TrainTotal),
                       'Performance_measures/Valid_Accuracy': (100*self.ValCorrect / self.ValTotal),
                       'Performance_measures/Balanced_Training_Accuracy': np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes,
                       'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})

            Performance_data = []
            Performance_data.append(TimeVec[Comp]+1)
            Performance_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            Grad_data = []
            Grad_data.append(TimeVec[Comp]+1)
            Grad_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            for i in range(0, self.model.num_classes):
                wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][Comp+1],
                           'Performance_measures/Valid_Loss_Class_{}'.format(i): self.model.ValidClassesLoss[i][Comp+1],
                           'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][Comp+1],
                           'Performance_measures/Valid_Accuracy_Class_{}'.format(i): self.model.ValidClassesAcc[i][Comp],
                           'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                           'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})    
                
                Performance_data.append(self.model.TrainClassesLoss[i][Comp+1])
                Performance_data.append(self.model.TrainClassesAcc[i][Comp+1])
            #if self.params['Dynamic']=='SGD':
            wandb.log({'Check/Step_Norm': (self.Step**0.5),
                       'Check/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
    
            if(all(self.GradCopy) or self.params['Dynamic']=='PCNSGD+R'): #if we are in the projection normalization mode we calculate with the whole dataset, so we are granted to have all the component 
                
                for k in range(0, self.model.num_classes):
                    wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(k): (self.model.StepGradientClassNorm[k][Comp+1] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})   
                    Grad_data.append(self.model.StepGradientClassNorm[k][Comp+1])
                
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    wandb.log({'GradientAngles/Gradient_Angles_Between_Classes_{}_and_{}'.format(i, j): (self.model.PCGAngles[AngleIndex][Comp+1] ),
                               #'GradientAngles/Gradient_Angles_NormComp_Between_Classes_{}_and_{}'.format(i, j): (self.model.GradAnglesNormComp[AngleIndex][Comp] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
                    Grad_data.append(self.model.PCGAngles[AngleIndex][Comp+1])
                    AngleIndex+=1 
                    
            self.Performance_data_table.add_data(*copy.deepcopy(Performance_data))
            self.Grad_data_table.add_data(*copy.deepcopy(Grad_data))
        
        elif self.params['ValidMode']=='Test':
            wandb.log({'Performance_measures/Training_Loss': np.array(self.model.TrainLoss)[-1],
                       'Performance_measures/Valid_Loss': np.array(self.model.ValidLoss)[-1],
                       'Performance_measures/Test_Loss': np.array(self.model.TestLoss)[-1],
                       'Performance_measures/Training_Accuracy': (100*self.TrainCorrect / self.TrainTotal),
                       'Performance_measures/Valid_Accuracy': (100*self.ValCorrect / self.ValTotal),
                       'Performance_measures/Test_Accuracy': (100*self.TestCorrect / self.TestTotal),
                       'Performance_measures/Balanced_Training_Accuracy': np.sum(self.model.TrainClassesAcc, axis=0)[Comp+1]/self.model.num_classes,
                       'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
            Performance_data = []
            Performance_data.append(TimeVec[Comp]+1)
            Performance_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            Grad_data = []
            Grad_data.append(TimeVec[Comp]+1)
            Grad_data.append(TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate'])
            for i in range(0, self.model.num_classes):
                wandb.log({'Performance_measures/Training_Loss_Class_{}'.format(i): self.model.TrainClassesLoss[i][Comp+1],
                           'Performance_measures/Test_Loss_Class_{}'.format(i): self.model.TestClassesLoss[i][Comp+1],
                           'Performance_measures/Valid_Loss_Class_{}'.format(i): self.model.ValidClassesLoss[i][Comp+1],
                           'Performance_measures/Training_Accuracy_Class_{}'.format(i): self.model.TrainClassesAcc[i][Comp+1],
                           'Performance_measures/Test_Accuracy_Class_{}'.format(i): self.model.TestClassesAcc[i][Comp],
                           'Performance_measures/Valid_Accuracy_Class_{}'.format(i): self.model.ValidClassesAcc[i][Comp],
                           'Performance_measures/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                           'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})    
                
                Performance_data.append(self.model.TrainClassesLoss[i][Comp+1])
                Performance_data.append(self.model.TrainClassesAcc[i][Comp+1])
            #if self.params['Dynamic']=='SGD':
            wandb.log({'Check/Step_Norm': (self.Step**0.5),
                       'Check/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                       'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
    
            if(all(self.GradCopy) or self.params['Dynamic']=='PCNSGD+R'): #if we are in the projection normalization mode we calculate with the whole dataset, so we are granted to have all the component 
                
                for k in range(0, self.model.num_classes):
                    wandb.log({'GradientAngles/Gradient_Single_batch_Norm_of_Classes_{}'.format(k): (self.model.StepGradientClassNorm[k][Comp+1] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})   
                    Grad_data.append(self.model.StepGradientClassNorm[k][Comp+1])
                
                AngleIndex=0
                for i,j in((i,j) for i in range(self.model.num_classes) for j in range(i)):
                    wandb.log({'GradientAngles/Gradient_Angles_Between_Classes_{}_and_{}'.format(i, j): (self.model.PCGAngles[AngleIndex][Comp+1] ),
                               #'GradientAngles/Gradient_Angles_NormComp_Between_Classes_{}_and_{}'.format(i, j): (self.model.GradAnglesNormComp[AngleIndex][Comp] ),
                               'GradientAngles/Rescaled_Steps': (TimeVec[Comp]*self.params['batch_size']/self.params['learning_rate']),
                               'Performance_measures/True_Steps_+_1': TimeVec[Comp]+1})
                    Grad_data.append(self.model.PCGAngles[AngleIndex][Comp+1])
                    AngleIndex+=1 

                self.Performance_data_table.add_data(*copy.deepcopy(Performance_data))
                self.Grad_data_table.add_data(*copy.deepcopy(Grad_data))

        
    def Gradient_Norms_logs(self, TimeComp):
        """
        here we log the norm of the gradient (total and per-class) in wandb

        Parameters
        ----------
        TimeComp : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        wandb.log({'Check/Gradient_Total_Norm': self.total_norm,
                   'Check/Epoch': self.params['epoch']})        
        for i in range(0, self.model.num_classes):
            wandb.log({'Check/Mean_Gradient_Norm_OfClass_{}'.format(i):  self.model.ClassesGradientNorm[TimeComp][i],
                       'Check/Epoch': self.params['epoch']})       
            

    def Histo_logs(self, n0, n1, n1_n0):
        """
        Here we save, as a sanity check the number of class element in each batch and their ratio.
        for now we collect them all togheter, but it is possible to save the histograms epochs bt epochs to see how does it change

        Parameters
        ----------
        n0 : TYPE
            DESCRIPTION.
        n1 : TYPE
            DESCRIPTION.
        n1_n0 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        """
        print(n0)
        print('LA RESA DEI CONTI')
        print(np.array(n0))
        print(np.array(n1))
        print(np.array(n1_n0))
        
        hist_n0 = np.histogram(np.array(n0), density=True, bins = 'auto')
        hist_n1 = np.histogram(np.array(n1), density=True, bins = 'auto')            
        hist_n1_n0 = np.histogram(np.array(n1_n0), density=True, bins = 'auto')  
        
        wandb.log({"Histo/n0": wandb.Histogram(np_histogram=hist_n0)})
        wandb.log({'Histo/n1': wandb.Histogram(np_histogram=hist_n1)})
        wandb.log({'Histo/n1_n0': wandb.Histogram(np_histogram=hist_n1_n0)})
        """

        
        n0_table = wandb.Table(data=n0, columns=["n0"])
        n1_table = wandb.Table(data=n1, columns=["n1"])       
        #n1_n0_table = wandb.Table(data=n1_n0, columns=["n1_n0"])    
        
        histogram_n0 = wandb.plot.histogram(n0_table, value='n0', title='Histogram_n0')
        histogram_n1 = wandb.plot.histogram(n1_table, value='n1', title='Histogram_n1')
        #histogram_n1_n0 = wandb.plot.histogram(n1_n0_table, value='n1_n0', title='Ratio_Histogram')
        
        wandb.log({'Classes_Presence_Histo/histogram_n0': histogram_n0, 
                   'Classes_Presence_Histo/histogram_n1': histogram_n1 
                   #,'Classes_Presence_Histo/histogram_n1_n0': histogram_n1_n0
                   })
        
        
        
        #saving matplotlib plot
        #we first convert list into numpy array, we flatten it and finally we plot the histo and log it
        n0_arr = (np.array(n0)).flatten()
        n1_arr = (np.array(n1)).flatten()        
        #n1_n0_arr = (np.array(n1_n0)).flatten()    
        kwargs = dict(histtype='stepfilled', alpha=0.3, bins='auto',density=True)
        
        plt.hist(n0_arr, **kwargs, label = r'$n_0$')
        plt.title(r"$n_0$ Distribution")
        plt.legend(loc='best', fontsize=7)
        wandb.log({"Classes_Presence_Histo/histo_n0": wandb.Image(plt)})
        plt.show()
        plt.clf()
        
        plt.hist(n1_arr, **kwargs, label = r'$n_1$')
        plt.title(r"$n_1$ Distribution")
        plt.legend(loc='best', fontsize=7)
        wandb.log({"Classes_Presence_Histo/histo_n1": wandb.Image(plt)})
        plt.show()
        plt.clf()
        
        """
        plt.hist(n1_n0_arr, **kwargs, label = r'$\frac{n_1}{n_0}$')
        plt.title(r"$\frac{n_1}{n_0}$ Distribution")
        plt.legend(loc='best', fontsize=7)
        wandb.log({"Classes_Presence_Histo/histo_n1_n0": wandb.Image(plt)}) 
        plt.show()  
        plt.clf()            
        """
        
        """
        hist_n0 = np.array(n0)
        hist_n1 = np.array(n1)            
        hist_n1_n0 = np.array(n1_n0)  
        
        
        wandb.log({"Histo/n0": wandb.Histogram(hist_n0)})
        wandb.log({'Histo/n1': wandb.Histogram(hist_n1)})
        wandb.log({'Histo/n1_n0': wandb.Histogram(hist_n1_n0)})
        """


        
    def CustomizedX_Axis(self):
        """
        Set the default x axis to assign it on each group of logged measures

        Returns
        -------
        None.

        """
        wandb.define_metric("MeanSquare_Distance/t")
        # set all other MeanSquare_Distance/ metrics to use this step
        wandb.define_metric("MeanSquare_Distance/*", step_metric="MeanSquare_Distance/t")

        wandb.define_metric("Overlap/t")
        # set all other Overlap/ metrics to use this step
        wandb.define_metric("Overlap/*", step_metric="Overlap/t")        
        
        wandb.define_metric("Performance_measures/True_Steps_+_1")
        # set all other MeanSquare_Distance/ metrics to use this step
        wandb.define_metric("Performance_measures/*", step_metric="Performance_measures/True_Steps_+_1")    
        
        wandb.define_metric("Grad_Overlap/*", step_metric="Grad_Overlap/Steps")
        
        
        wandb.define_metric("Check/Epoch")
        # set all other MeanSquare_Distance/ metrics to use this step
        wandb.define_metric("Check/*", step_metric="Check/Epoch")         
        wandb.define_metric("Check/Step_Norm", step_metric="Check/True_Steps_+_1")  
        wandb.define_metric("GradientAngles/*", step_metric="GradientAngles/True_Steps_+_1")  


    def RAM_check(self,line_number):
        """
        print the RAM IN Gb on a file

        Parameters
        ----------
        line_number : line number where the function is called
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print("the amount of RAM used in line ~~{}~~ of the code MainBlock.py (PID: {}) is:  ".format(line_number, os.getpid()) , psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, flush = True, file = self.params['memory_leak_file_object']) 


    def LineNumber(self):
        """
        return the line number of the code

        Returns
        -------
        None.

        """
        cf = currentframe()
        return cf.f_back.f_lineno        

    def TorchCheckpoint(self):
        """
        Create/update the checkpoint of the status

        Returns
        -------
        None.

        """
        #TODO: VERIFY THAT THE SAVING AND LOAD WORKS PROPERLY (WE MODIFIED THE MODEL, CHARGING VARIABÒES ON IT)
        #WARNING: for some models a snapshot of the last epoch could not be enough; there could be a dependence also from the previous state of the net 
        #saving the model () at the end of the run with
        torch.save({'model_state_dict' : self.model.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(), #When saving a general checkpoint, you must save more than just the model’s state_dict. It is important to also save the optimizer’s state_dict, as this contains buffers and parameters that are updated as the model trains.
                    'epoch': self.params['epoch'],  #Other items that you may want to save are the epoch at which you stopped
                    'step': self.params['IterationCounter'],  #or number of iteration
                    'OldNSteps': self.params['NSteps'],
                    'TimeComp': self.params['TimesComponentCounter'],  #or number of evaluation block encountered
                    #TODO: AGGIUNGI I TEMPI PER IL CALCOLO DELLE FUNZIONI DI CORRELAZIONE (Tw)
                    'proj_id': self.params['ProjId'],
                    #metrix saved at the latest component updated
                    'OldTP':self.model.TP[:,:self.params['TimesComponentCounter']],
                    'OldTN':self.model.TN[:,:self.params['TimesComponentCounter']],
                    'OldFN':self.model.FN[:,:self.params['TimesComponentCounter']],
                    'OldFP':self.model.FP[:,:self.params['TimesComponentCounter']],
                    'OldTotP':self.model.TotP[:,:self.params['TimesComponentCounter']],
                    'OldTimeVector': self.Times,
                    'OldTrainPrec': self.model.Prec[:,:self.params['TimesComponentCounter']],
                    'OldTrainRecall': self.model.Recall[:,:self.params['TimesComponentCounter']],
                    'OldTrainF_Measure':self.model.FMeasure[:,:self.params['TimesComponentCounter']],
                    'OldPCGAngles':self.model.PCGAngles[:,:self.params['TimesComponentCounter']+1],
                    'OldTrainAngles':self.model.TrainAngles[:,:self.params['TimesComponentCounter']+1],
                    'OldTestAngles':self.model.TestAngles[:,:self.params['TimesComponentCounter']],
                    'OldRepresentationClassesNorm':self.model.RepresentationClassesNorm[:,:self.params['TimesComponentCounter']+1],
                    'OldClassesGradientNorm':self.model.ClassesGradientNorm[:self.params['n_epochs'],:],
                    'OldGradAnglesNormComp':self.model.GradAnglesNormComp[:,:self.params['TimesComponentCounter']],
                    'OldStepGradientClassNorm':self.model.StepGradientClassNorm[:,:self.params['TimesComponentCounter']],
                    
                    'OldTrainLoss':self.model.TrainLoss,#unidimensional measure are stored in list over which we append time by time
                    'OldTrainAcc':self.model.TrainAcc,
                    'OldTrainClassesLoss':self.model.TrainClassesLoss[:,:self.params['TimesComponentCounter']+1],
                    'OldTrainClassesAcc':self.model.TrainClassesAcc[:,:self.params['TimesComponentCounter']+1],
                    'OldValidLoss':self.model.ValidLoss,
                    'OldValidAcc':self.model.ValidAcc,
                    'OldValidClassesLoss':self.model.ValidClassesLoss[:,:self.params['TimesComponentCounter']+1],
                    'OldValidClassesAcc':self.model.ValidClassesAcc[:,:self.params['TimesComponentCounter']+1],
                        #we save always also the test measure, then depending if we are in the test or valid mode they could be meaningful or not
                    'OldTestLoss':self.model.TestLoss,
                    'OldTestAcc':self.model.TestAcc,
                    'OldTestClassesLoss':self.model.TestClassesLoss[:,:self.params['TimesComponentCounter']+1],
                    'OldTestClassesAcc':self.model.TestClassesAcc[:,:self.params['TimesComponentCounter']+1],                                   
                    }, self.params['FolderPath'] +'/model.pt')
        #just after the update of pytorch model we update the version stored in wandb
        wandb.save(self.params['FolderPath'] +'/model.pt') #save the model at the end of simulation to restart it from the end point         


    def RecallOldVariables(self, checkpoint):
        """
        This method is called when we restart an old simulation from its checkpoint; we store the variables collected in that simulation from the old files.
        The variables are stored in the model.pt file (checkpoint file) so that you don't need the variables file to recall it.
        
        The 1-d variables (like TrainLoss) are stored in list; we can then just assign the old variable to the new variable and keep appending new measures
        The multidimensional variables (like Classes measures) are stored in numpy array; in this case we can merge the old variables and new initialized vector using numpy.concatenate((a1, a2, ...), axis=0, out=None, dtype=None, casting="same_kind")
        
        NOTE: during checkpoint all vector are stored as they are, without any transposistion or modification in their shape, so you don't have to apport any changes for the merging
        Parameters
        ----------
        checkpoint : dict 
            dict of variables stored from checkpoint of previous simulation
        Returns
        -------
        None.
        """
        
        self.model.TP = np.concatenate((checkpoint['OldTP'],self.model.TP), axis=1)
        self.model.TN = np.concatenate((checkpoint['OldTN'],self.model.TN), axis=1)
        self.model.FP = np.concatenate((checkpoint['OldFP'],self.model.FP), axis=1)
        self.model.FN = np.concatenate((checkpoint['OldFN'],self.model.FN), axis=1)
        self.model.TotP = np.concatenate((checkpoint['OldTotP'],self.model.TotP), axis=1)
        self.model.GradAnglesNormComp = np.concatenate((checkpoint['OldGradAnglesNormComp'],self.model.GradAnglesNormComp), axis=1)
        self.model.StepGradientClassNorm = np.concatenate((checkpoint['OldStepGradientClassNorm'],self.model.StepGradientClassNorm), axis=1)
        
        self.model.Prec = np.concatenate((checkpoint['OldTrainPrec'],self.model.Prec), axis=1)
        self.model.Recall = np.concatenate((checkpoint['OldTrainRecall'],self.model.Recall), axis=1)
        self.model.FMeasure = np.concatenate((checkpoint['OldTrainF_Measure'],self.model.FMeasure), axis=1)
        self.model.PCGAngles = np.concatenate((checkpoint['OldPCGAngles'],self.model.PCGAngles), axis=1)
        self.model.TrainAngles = np.concatenate((checkpoint['OldTrainAngles'],self.model.TrainAngles), axis=1)
        self.model.TestAngles = np.concatenate((checkpoint['OldTestAngles'],self.model.TestAngles), axis=1)
        self.model.RepresentationClassesNorm = np.concatenate((checkpoint['OldRepresentationClassesNorm'],self.model.RepresentationClassesNorm), axis=1)
        self.model.ClassesGradientNorm = np.concatenate((checkpoint['OldClassesGradientNorm'],self.model.ClassesGradientNorm), axis=0) #note that here we concatenate along axis 0 because order fo classes and times are reversed
        self.model.TrainLoss = checkpoint['OldTrainLoss'] #for list variable we don't need to concatenate, since there is no initaliazed vector (we build it concatenating measures time by time)
        self.model.TrainAcc = checkpoint['OldTrainAcc']
        self.model.TrainClassesLoss = np.concatenate((checkpoint['OldTrainClassesLoss'],self.model.TrainClassesLoss), axis=1)
        self.model.TrainClassesAcc = np.concatenate((checkpoint['OldTrainClassesAcc'],self.model.TrainClassesAcc), axis=1)
        self.model.ValidLoss = checkpoint['OldValidLoss']
        self.model.ValidAcc = checkpoint['OldValidAcc']
        self.model.ValidClassesLoss = np.concatenate((checkpoint['OldValidClassesLoss'],self.model.ValidClassesLoss), axis=1)
        self.model.ValidClassesAcc = np.concatenate((checkpoint['OldValidClassesAcc'],self.model.ValidClassesAcc), axis=1)
        self.model.TestLoss = checkpoint['OldTestLoss']
        self.model.TestAcc = checkpoint['OldTestAcc']
        self.model.TestClassesLoss = np.concatenate((checkpoint['OldTestClassesLoss'],self.model.TestClassesLoss), axis=1)
        self.model.TestClassesAcc = np.concatenate((checkpoint['OldTestClassesAcc'],self.model.TestClassesAcc), axis=1)










