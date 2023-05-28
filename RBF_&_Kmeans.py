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

# RBF (Radial Basis Function) Class Code #######################################
import numpy as np
#import pcn
#import kmeans

class rbf:
    """ The Radial Basis Function network
    Parameters are number of RBFs, and their width, how to train the network 
    (pseudo-inverse or kmeans) and whether the RBFs are normalised"""

    def __init__(self,inputs,targets,nRBF,sigma=0,usekmeans=0,normalise=0):
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nRBF = nRBF
        self.usekmeans = usekmeans
        self.normalise = normalise
        
        if usekmeans:
            self.kmeansnet = kmeans.kmeans(self.nRBF,inputs)
            
        self.hidden = np.zeros((self.ndata,self.nRBF+1))
        
        if sigma==0:
            # Set width of Gaussians
            d = (inputs.max(axis=0)-inputs.min(axis=0)).max()
            self.sigma = d/np.sqrt(2*nRBF)  
        else:
            self.sigma = sigma
                
        self.perceptron = pcn.pcn(self.hidden[:,:-1],targets)
        
        # Initialise network
        self.weights1 = np.zeros((self.nin,self.nRBF))
        
    def rbftrain(self,inputs,targets,eta=0.25,niterations=100):
                
        if self.usekmeans==0:
            # Version 1: set RBFs to be datapoints
            indices = range(self.ndata)
            np.random.shuffle(indices)
            for i in range(self.nRBF):
                self.weights1[:,i] = inputs[indices[i],:]
        else:
            # Version 2: use k-means
            self.weights1 = np.transpose(self.kmeansnet.kmeanstrain(inputs))

        for i in range(self.nRBF):
            self.hidden[:,i] = np.exp(-np.sum((inputs - np.ones((1,self.nin))*self.weights1[:,i])**2,axis=1)/(2*self.sigma**2))
        if self.normalise:
            self.hidden[:,:-1] /= np.transpose(np.ones((1,np.shape(self.hidden)[0]))*self.hidden[:,:-1].sum(axis=1))
        
        # Call Perceptron without bias node (since it adds its own)
        self.perceptron.pcntrain(self.hidden[:,:-1],targets,eta,niterations)
        
    def rbffwd(self,inputs):

        hidden = np.zeros((np.shape(inputs)[0],self.nRBF+1))

        for i in range(self.nRBF):
            hidden[:,i] = np.exp(-np.sum((inputs - np.ones((1,self.nin))*self.weights1[:,i])**2,axis=1)/(2*self.sigma**2))

        if self.normalise:
            hidden[:,:-1] /= np.transpose(ones((1,np.shape(hidden)[0]))*hidden[:,:-1].sum(axis=1))
        
        # Add the bias
        hidden[:,-1] = -1

        outputs = self.perceptron.pcnfwd(hidden)
        return outputs
    
    def confmat(self,inputs,targets):
        """Confusion matrix"""

        outputs = self.rbffwd(inputs)
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
# RBF (Radial Basis Function) Class Code End ###################################

# Kmeans Class Code ############################################################
import numpy as np

class kmeans:
	""" The k-Means algorithm"""
	def __init__(self,k,data):

		self.nData = np.shape(data)[0]
		self.nDim = np.shape(data)[1]
		self.k = k
		
	def kmeanstrain(self,data,maxIterations=10):
		
		# Find the minimum and maximum values for each feature
		minima = data.min(axis=0)
		maxima = data.max(axis=0)
	
		# Pick the centre locations randomly
		self.centres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
		oldCentres = np.random.rand(self.k,self.nDim)*(maxima-minima)+minima
	
		count = 0
		#print centres
		while np.sum(np.sum(oldCentres-self.centres))!= 0 and count<maxIterations:
	
			oldCentres = self.centres.copy()
			count += 1
	
			# Compute distances
			distances = np.ones((1,self.nData))*np.sum((data-self.centres[0,:])**2,axis=1)
			for j in range(self.k-1):
				distances = np.append(distances,np.ones((1,self.nData))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
	
			# Identify the closest cluster
			cluster = distances.argmin(axis=0)
			cluster = np.transpose(cluster*np.ones((1,self.nData)))
	
			# Update the cluster centres	
			for j in range(self.k):
				thisCluster = np.where(cluster==j,1,0)
				if sum(thisCluster)>0:
					self.centres[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
			#plot(data[:,0],data[:,1],'kx')
			#plot(centres[:,0],centres[:,1],'ro')
		return self.centres
	
	def kmeansfwd(self,data):
		
		nData = np.shape(data)[0]
		# Compute distances
		distances = np.ones((1,nData))*np.sum((data-self.centres[0,:])**2,axis=1)
		for j in range(self.k-1):
			distances = np.append(distances,np.ones((1,nData))*np.sum((data-self.centres[j+1,:])**2,axis=1),axis=0)
	
		# Identify the closest cluster
		cluster = distances.argmin(axis=0)
		cluster = np.transpose(cluster*np.ones((1,nData)))
	
		return cluster
# Kmeans Class Code End ########################################################

# Google Drive mount to import dataset
from google.colab import drive
drive.mount('/content/drive')

import numpy as np    # matrix, process the data
import pandas as pd   # dataframes, import the data
import pylab as pl    # visualization
import seaborn as sns # statistical visualization


# Get the datasets into a Pandas dataframe
datatraining_df = pd.read_csv('/content/datatest.txt').iloc[1:,:]
datatest_df = pd.read_csv('/content/datatest2.txt').iloc[1:,:]

# Save dataframes into numpy matrices
train = datatraining_df.to_numpy()
test = datatest_df.to_numpy()

print "Data validation:", type(train), train.shape
print "Data testing:", type(test), test.shape

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

# Radial Basis Functions (RBF)

import numpy as np

iris = np.loadtxt('../3 MLP/iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),iris.min(axis=0)*np.ones((1,5))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]
#print iris[0:5,:]

#target = zeros((shape(iris)[0],2));
#indices = where(iris[:,4]==0) 
#target[indices,0] = 1
#indices = where(iris[:,4]==1)
#target[indices,1] = 1
#indices = where(iris[:,4]==2)
#target[indices,0] = 1
#target[indices,1] = 1

target = np.zeros((np.shape(iris)[0],3));
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1
indices = np.where(iris[:,4]==1)
target[indices,1] = 1
indices = np.where(iris[:,4]==2)
target[indices,2] = 1


order = range(np.shape(iris)[0])
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]

#print train.max(axis=0), train.min(axis=0)

import rbf
net = rbf.rbf(train,traint,5,1,1)

net.rbftrain(train,traint,0.25,2000)
#net.confmat(train,traint)
net.confmat(test,testt)