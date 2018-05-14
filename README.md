# About
This project implements the method of 'Bag of Visual Words' to classify images into vehicles and non-vehicles from the available dataset.


# Methodology
Following are the steps implemented (in order):
1. Split the dataset into Training and Test (already divided the dataset in the respective folders)
1. Extract SIFT features from the training dataset
1. Learn a codebook of size 1000 i.e. the size of Bag of Visual Words is 1000, using K-Means clustering
1. Train a linear, soft-margin SVM using 5-fold cross-validation on the training dataset
1. Test the accuracy of the SVM model learnt on the test dataset
