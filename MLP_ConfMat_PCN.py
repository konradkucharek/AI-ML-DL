# -*- coding: utf-8 -*-

# To check Python Version
import sys

print(sys.version)
# sys.version should be 2.7.17

# MLP Class Code ##############################################################
import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print count
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print "Stopped", new_val_error,old_val_error1, old_val_error2
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print "Iteration: ",n, " Error: ",error    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print "error"
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 = self.weights1 - updatew1
            self.weights2 = self.weights2 - updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print "error"

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print "Confusion matrix is:"
        print cm
        print "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100
# MLP Class Code End ###########################################################

# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import pylab as pl
import numpy as np

# PCN (Perceptron) Class Code ##################################################

class pcn:
	""" A basic Perceptron"""
	
	def __init__(self,inputs,targets):
		""" Constructor """
		# Set up network size
		if np.ndim(inputs)>1:
			self.nIn = np.shape(inputs)[1]
		else: 
			self.nIn = 1
	
		if np.ndim(targets)>1:
			self.nOut = np.shape(targets)[1]
		else:
			self.nOut = 1

		self.nData = np.shape(inputs)[0]
	
		# Initialise network
		self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

	def pcntrain(self,inputs,targets,eta,nIterations):
		""" Train the thing """	
		# Add the inputs that match the bias node
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		# Training
		change = range(self.nData)

		for n in range(nIterations):
			
			self.activations = self.pcnfwd(inputs);
			self.weights -= eta*np.dot(np.transpose(inputs),self.activations-targets)
		
			# Randomise order of inputs
			#np.random.shuffle(change)
			#inputs = inputs[change,:]
			#targets = targets[change,:]
			
		#return self.weights

	def pcnfwd(self,inputs):
		""" Run the network forward """
		# Compute activations
		activations =  np.dot(inputs,self.weights)

		# Threshold the activations
		return np.where(activations>0,1,0)


	def confmat(self,inputs,targets):
		"""Confusion matrix"""

		# Add the inputs that match the bias node
		inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		
		outputs = np.dot(inputs,self.weights)
	
		nClasses = np.shape(targets)[1]

		if nClasses==1:
			nClasses = 2
			outputs = np.where(outputs>0,1,0)
		else:
			# 1-of-N encoding
			outputs = np.argmax(outputs,1)
			targets = np.argmax(targets,1)

		cm = np.zeros((nClasses,nClasses))
		for i in range(nClasses):
			for j in range(nClasses):
				cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

		print cm
		print np.trace(cm)/np.sum(cm)
		
def logic():
	import pcn
	""" Run AND and XOR logic functions"""

	a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
	b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

	p = pcn.pcn(a[:,0:2],a[:,2:])
	p.pcntrain(a[:,0:2],a[:,2:],0.25,10)
	p.confmat(a[:,0:2],a[:,2:])

	q = pcn.pcn(b[:,0:2],b[:,2:])
	q.pcntrain(b[:,0:2],b[:,2:],0.25,10)
	q.confmat(b[:,0:2],b[:,2:])


# PCN (Perceptron) Class Code End ##############################################

# Google Drive mount to import dataset
from google.colab import drive
drive.mount('/content/drive')

import numpy as np    # matrix, process the data
import pandas as pd   # dataframes, import the data
import pylab as pl    # visualization
import seaborn as sns # statistical visualization


# Get the datasets into a Pandas dataframe
datatraining_df = pd.read_csv('/content/datatraining.txt').iloc[1:,:]
datatest_df = pd.read_csv('/content/datatest2.txt').iloc[1:,:]

# Save dataframes into numpy matrices
train = datatraining_df.to_numpy()
test = datatest_df.to_numpy()

print "Data training:", type(train), train.shape
print "Data test:", type(test), test.shape

# change all matrices to be the same size 
import random
randomList = []
rowsTrain, colsTrain = train.shape
rowsTest, colsTest = test.shape

for i in range (0, (rowsTest - rowsTrain)):
    n = random.randint(0, rowsTrain)
    test = np.delete(test, (n), axis=0)

# Split data and target
trainD = train[:, 1:6] # all rows, except date and occupancy columns 
trainT = train[:, 6:7] # all rows, occupancy column  
testD = test[:, 1:6]
testT = test[:, 6:7]

# change all datatypes to float
trainD = trainD.astype(np.float)
trainT = trainT.astype(np.float)
testD = testD.astype(np.float)
testT = testT.astype(np.float)

print "Train data and target:", trainD.shape, trainT.shape
print "Test data and target:", testD.shape, testT.shape

# PCN (Non-Normalized)
import pcn            

# Perceptron training on the original dataset
print "Output on original data"
p = pcn.pcn(trainD,trainT)
p.pcntrain(trainD, trainT, 0.25, 100)
p.confmat(trainD,trainT)

# Test
nData = np.shape(trainD)[0]
testin = np.concatenate((trainD,-np.ones((nData,1))),axis=1)
testout = p.pcnfwd(testin)


# Error Calculation and Display
error_PCN_raw = np.sum((testout - testT)**2)
print "PCN Error (Non-Normalized): ", error_PCN_raw

# MLP (Non-Normalized)
print "Train max: ", trainD.max(axis=0)
print "Train min: ", trainD.min(axis=0)

# Train the network
import mlp
net = mlp.mlp(trainD,trainT,5,outtype='logistic')
net.mlptrain(trainD, trainT, 0.25, 100)
net.confmat(testD,testT)

# Test
nData = np.shape(trainD)[0]
testin = np.concatenate((trainD,-np.ones((nData,1))),axis=1)
testout = net.mlpfwd(testin)

# Error Calculation and Display
error_MLP_raw = np.sum((testout - testT)**2)
print "MLP Error (Non-Normalized): ", error_MLP_raw

# Normalization of Dataset
trainNormal = (train[:,1:] - train[:,1:].mean(axis=0))/train[:,1:].var(axis=0)
testNormal = (test[:,1:] - test[:,1:].mean(axis=0))/test[:,1:].var(axis=0)

print "Normalized Data Training:", trainNormal.shape
print "\n", trainNormal
print "\n\nNormalized Data Test:", testNormal.shape
print "\n", testNormal

# PCN (Normalized)
p = pcn.pcn(trainND,trainNT)
p.pcntrain(trainND, trainNT, 0.25, 100)
p.confmat(trainND,trainNT)

# Test
nData = np.shape(trainND)[0]
testin = np.concatenate((trainND,-np.ones((nData,1))),axis=1)
testout = p.pcnfwd(testin)

# Error Calculation and Display
error_PCN_norm = np.sum((testout - testNT)**2)
print "PCN Error (Normalized) : ", error_PCN_norm

# MLP (Normalized)

print "Train max: ", trainD.max(axis=0)
print "Train min: ", trainD.min(axis=0)
print "\nMLP on normalized dataset"

# Train the network
import mlp
net = mlp.mlp(trainND,trainNT,5,outtype='logistic')
net.mlptrain(trainND, trainNT, 0.25, 100)
net.confmat(testND,testNT)

# Test
nData = np.shape(trainND)[0]
testin = np.concatenate((trainND,-np.ones((nData,1))),axis=1)
testout = net.mlpfwd(testin)
error_MLP_normal = np.sum((testout - testNT)**2)
print "MLP Error (Normalized) : ", error_MLP_normal