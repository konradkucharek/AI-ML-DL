# -*- coding: utf-8 -*-

# To check Python Version
import sys

print(sys.version)
# sys.version should be 2.7.17

import pylab as pl
import numpy as np

# Routine for opening file (from anywhere locally) to be used with the program
from google.colab import files
 
pima = files.upload()

# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

# Linear Regressor Code #######################################################

def linreg(inputs,targets):

	inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
	beta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(inputs),inputs)),np.transpose(inputs)),targets)

	outputs = np.dot(inputs,beta)
	#print shape(beta)
	#print outputs
	return beta

# Linear Regressor Code End ###################################################

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

# PCN (Non-Normalized)
pima = np.loadtxt('winequality-red.csv',delimiter=';')

# Perceptron training on the original dataset
p = pcn(pima[:,:11],pima[:,11:12])
p.pcntrain(pima[:,:11],pima[:,11:12],0.25,100)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
p.pcnfwd(pima[:,:12])

trainin = pima[::2,:11]
testin = pima[1::2,:11]
traintgt = pima[::2,11:12]
testtgt = pima[1::2,11:12]

# Perceptron training on the preprocessed dataset
p1 = pcn(trainin,traintgt)
p1.pcntrain(trainin,traintgt,0.25,100)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)

testout = p1.pcnfwd(testin)

# Error Calculation and Display
error = np.sum((testout - testtgt)**2)
print("PCN Error (Non-Normalized) : " +str(error))

# PCN (Normalized)
pima = np.loadtxt('winequality-red.csv', delimiter=';', skiprows=1)

dataset = np.array(pima)
normalized_dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
pima = normalized_dataset

# Perceptron training on the original dataset
p = pcn(pima[:,:11],pima[:,11:12])
p.pcntrain(pima[:,:11],pima[:,11:12],0.25,100)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
p.pcnfwd(pima[:,:12])

trainin = pima[::2,:11]
testin = pima[1::2,:11]
traintgt = pima[::2,11:12]
testtgt = pima[1::2,11:12]

# Perceptron training on the preprocessed dataset
p1 = pcn(trainin,traintgt)
p1.pcntrain(trainin,traintgt,0.25,100)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)

testout = p1.pcnfwd(testin)

# Error Calculation and Display
error = np.sum((testout - testtgt)**2)
print("PCN Error (Normalized) : " +str(error))

# Linear Regressor (Non-Normalized)

pima = np.loadtxt('winequality-red.csv', delimiter=';', skiprows=1)

#split the dataset into a training and testing set
testin = pima[1::2,:11]
trainin = pima[::2,:11]
traintgt = pima[::2,11:12]
testtgt = pima[1::2,11:12]

#call the linear regressor and train the data
beta = linreg(trainin,traintgt)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
testout = np.dot(testin,beta)

# Error Calculation
error = np.sum((testout - testtgt)**2)

# Error Display
print 'Linear Regressor Error (Non-Normalized) :' , error

# Linear Regressor (Normalized)

pima = np.loadtxt('winequality-red.csv', delimiter=';', skiprows=1)

#normalize the data before splitting
dataset = np.array(pima)
normalized_dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
pima = normalized_dataset

#split the dataset
testin = pima[1::2,:11]
trainin = pima[::2,:11]
traintgt = pima[::2,11:12]
testtgt = pima[1::2,11:12]

#train the normalized data
beta = linreg(trainin,traintgt)
testin = np.concatenate((testin,-np.ones((np.shape(testin)[0],1))),axis=1)
testout = np.dot(testin,beta)

# Error Calculation
error = np.sum((testout - testtgt)**2)

# Error Display
print 'Linear Regressor Error (Normalized) :' , error
