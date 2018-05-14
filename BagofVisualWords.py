
# coding: utf-8

# In[9]:


### Author: Aditya Jain #####
### Topic: Classification based on Bag-of-Visual-Words Model ###
### Start Date: 22nd April, 2018 ###

import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from __future__ import division
import glob
import time
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import time

start_time = time.time()

featureDesList = []

# This function iterates over all the training images and returns a concatented feature descriptor list
def findFeaturesDes():
    listDes = []
    label = []    # 1 for vehicle, 0 for non-vehicle
    HistogramFeatures = []  # this stores features separately for the histogram generation
    i = 0
    
    TrainFolder = glob.glob("Training/*")
    
    for folder in TrainFolder:        
        VehFolder = glob.glob(folder + "/*")  
        
        for vehicle in VehFolder:
            images = glob.glob(vehicle + "/*")
            
            for img in images:                
                TrainImage = cv2.imread(img)               
                TrainImageGray = cv2.cvtColor(TrainImage,cv2.COLOR_BGR2GRAY)  # Converting to gray scale
                
                # Applying SIFT feature detector
                sift = cv2.SIFT(10)
                kp, des = sift.detectAndCompute(TrainImageGray,None)   # kp are the keypoints, des are the descriptors               
                
                if des is not None:
                    HistogramFeatures.append(des)                    
                    
                    if str(folder) == 'Training/non-vehicles':
                        label.append(0)                            
                    else:
                        label.append(1)
                    
                    # This stores all the features as a single entity
                    for descriptor in des:                        
                        listDes.append(descriptor)
                        
                            
                
                i += 1
                
                
    return listDes, i, label, HistogramFeatures
                
                
featureDesList, count, labels, histFeatures = findFeaturesDes()
print len(featureDesList)


# In[10]:


## Peforming K-means clustering to get bag of visual words

# kmeans = MiniBatchKMeans(n_clusters=1000).fit(featureDesList)
kmeans = KMeans(n_clusters=1000, random_state=0, n_init=3, max_iter=50).fit(featureDesList)
error = (kmeans.inertia_)/(len(featureDesList))  # distance error per feature vector


# In[11]:


error = (kmeans.inertia_)/(len(featureDesList))  # distance error per feature vector
print np.sqrt(error)


# In[ ]:


print np.shape(kmeans.cluster_centers_)


# In[ ]:


### Building the histogram for training

# This function returns the histogram with the labels for training
def histogramVector(featureSet, ClusCen, NoC):
    
    MegaHistogram = []      # This contains the histogram vector for all the images
    
    for image in featureSet:   
        
        histogram = np.zeros(NoC)    # histogram initialised
        
        for features in image:
        
            MinDis = 10000000;
            MinClusCenIndex = -10000;   
            i = 0;
            # finding the closest cluster center with the feature vector
            for center in ClusCen:          
                Dist = np.linalg.norm(features-center)   # finding euclidean distance between the feature and cluster center 
            
                if Dist < MinDis:
                    MinDis = Dist
                    MinClusCenIndex = i
                
                
                i += 1
                
            np.add.at(histogram, [MinClusCenIndex], 1)  
            
        MegaHistogram.append(histogram)
        
    return MegaHistogram


HistFeatureList = histogramVector(histFeatures, kmeans.cluster_centers_, 1000)
        
end_time = time.time()

print "Time Taken:", end_time-start_time


# In[ ]:


### Applying SVM Model with cross-validation
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

X_train = HistFeatureList
Y_train = labels
RegularParam = []
MeanScore = []
for i in range(1,20): 
    print i
    clf = svm.LinearSVC(C=i)
    scores = cross_val_score(clf, X_train, Y_train, cv=5)
    RegularParam.append(i)
    MeanScore.append(scores.mean()*100)
    
# Plotting
plt.plot(RegularParam, MeanScore)
plt.ylabel('Accuracy in %')
plt.xlabel('Regularisation Parameter Value')
plt.title('Accuracy v/s Regularisation Parameter for SVM')


# In[ ]:


### Finally doing the training

# Training with the chosen regularisation parameter
lin_clf = svm.LinearSVC(C=1.0)
lin_clf.fit(X_train, Y_train) 


# In[ ]:


### Testing on the test data

# Building the histogram for the test dataset
def findFeaturesDesTest():
    
    ImageNum = [] # Will be used to check later that which images have failed the prediction
    label = []    # 1 for vehicle, 0 for non-vehicle
    HistogramFeatures = []  # this stores features separately for the histogram generation
    i = 0
    
    TrainFolder = glob.glob("Test/*")
    
    for folder in TrainFolder:        
        VehFolder = glob.glob(folder + "/*")  
        
        for vehicle in VehFolder:
            images = glob.glob(vehicle + "/*")
            
            for img in images:                
                TrainImage = cv2.imread(img)               
                TrainImageGray = cv2.cvtColor(TrainImage,cv2.COLOR_BGR2GRAY)  # Converting to gray scale
                
                # Applying SIFT feature detector
                sift = cv2.SIFT(10)
                kp, des = sift.detectAndCompute(TrainImageGray,None)   # kp are the keypoints, des are the descriptors               
                
                if des is not None:
                    HistogramFeatures.append(des)                    
                    
                    if str(folder) == 'Test/NonVehicles':
                        label.append(0)                            
                    else:
                        label.append(1) 
                        
                    ImageNum.append(img)
                            
                
                i += 1
                
                
    return i, label, HistogramFeatures, ImageNum

Count, LabelTest, FeatureListTest, ImageNames = findFeaturesDesTest()

HistFeatureListTest = histogramVector(FeatureListTest, kmeans.cluster_centers_, 1000)


# In[ ]:


print len(LabelTest), len(HistFeautureListTest)


# In[ ]:


### Testing on the test dataset
Predictions = lin_clf.predict(HistFeatureListTest)

def classificationAccuracy(pred, truth):    
    size = len(pred)
    count = 0
    fail = []   # Stores the list of failed predictions
    
    for i in range(size):
        if pred[i] == truth[i]:
            count += 1
        else:
            fail.append(i)
            
    return (count/size)*100, fail
   

Accuracy, FailCases = classificationAccuracy(Predictions, LabelTest)
print Accuracy


# In[ ]:


print FailCases


# In[ ]:


print ImageNames[21]
print ImageNames[396]
print ImageNames[842]
print ImageNames[1386]
print ImageNames[2727]
print ImageNames[3234]
print ImageNames[3359]
print ImageNames[1894]

